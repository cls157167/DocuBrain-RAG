import base64
import logging
import os
import re
from pathlib import Path
from typing import Tuple, Dict

from openai import OpenAI

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import FileProcessingError, LLMError, ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.MinioClient import minio_client


class MDImgNode(BaseNode):
    name = "MDImg"
    config=get_config()

    def process(self, state: ImportGraphState) -> ImportGraphState:

        self.log_step("step1","获取MD文件内容、MD文件的Path对象、MD文件关联的图片目录的Path对象（定位图片储存目录）")
        MD_content,md_path_obj,img_path_obj=self._get_content_MDObi_ImgObi(state)

        if not img_path_obj.exists():
            self.logger.info("未找到images目录，跳过图片处理流程，直接进行下一步")
            return state

        self.log_step("step2",f"扫描images目录{img_path_obj}，筛选出在MD文件中被引用的图片列表,并返回上下文")
        target_img_list=self._scan_and_filter_images(MD_content,img_path_obj,self.config.image_extensions)
        if not target_img_list:
            self.logger.info("未找到在MD文件中的图片引用")
            return state

        self.log_step("step3","根据图片列表信息（路径、上下文），调用大模型，生成图片摘要")
        img_summaries=self.get_img_summaries(target_img_list)

        self.log_step("step4","根据图片路径上传minio服务器，得到minio路径，更新md_content内容，更新state")
        new_md_content=self._upload_images_and_replaces_md_content(minio_client,target_img_list,img_summaries,MD_content)
        state["md_content"]=new_md_content

        new_md_path=self._backup_new_md_file(md_path_obj,new_md_content)
        file_title=Path(new_md_path).stem
        state["file_title"]=file_title
        state["pdf_path"]=new_md_path
        return state



    def _get_content_MDObi_ImgObi(self, state:ImportGraphState)->Tuple[str,Path,Path]:
        """
        :param state:
        :return: MD文件内容、MD文件Path对象、MD文件中关联的图片的Path对象（图片储存目录）
        """

        self.log_step("step1.1","获取MD文件的Path对象")
        md_path=state["md_path"]
        if not md_path:
            raise FileProcessingError("md文件路径不存在",node_name=self.name)
        md_path_obi=Path(md_path)

        self.log_step("step1.2", "获取MD文件内容")
        lines=[]
        try:
            with open(md_path,"r",encoding="utf-8") as f:
                for line in f:
                    lines.append(line.rstrip("\n"))
            md_content="\n".join(lines)
        except IOError as e:
            raise FileProcessingError(f"无法读取文件{md_path_obi},原因是{e}",node_name=self.name)

        self.log_step("step1.3", "获取MD文件内图片的Path对象")
        img_path_obj=md_path_obi.parent/"images"

        return md_content,md_path_obi,img_path_obj

    def _scan_and_filter_images(self,MD_content,img_path_obj,allowed_extensions)->list[Tuple[str,str,Tuple[str,str,str]]]:
        """
        筛选MD文件中引用的图片列表，将图片和上下文一起返回供大模型识别生成摘要
        :param MD_content:
        :param img_path_obj:
        :param allowed_extensions:
        :return:【“图片名称”，“图片路径”，（图片前一个标题，上文，下文）】
        """
        target_img_list=[]

        self.log_step("step2.1","获取图片文件名")
        #获取图片目录下所有的图片文件名
        img_name_list=os.listdir(img_path_obj)
        for img_name in img_name_list:
            #检查文件拓展名是否在允许列表中
            if os.path.splitext(img_name)[1].lower() not in allowed_extensions:
                continue

            self.log_step("step2.2", "获取图片路径")
            img_full_path=str(img_path_obj/img_name)

            self.log_step("step2.3", "获取图片上下文")
            content_tuple=self._find_image_contexts_in_md(img_name,MD_content)
            if not content_tuple:
                self.logger.debug(f"图片{img_name}未在文中引用,跳过")
                continue

            primary_context=content_tuple

            target_img_list.append((img_name,img_full_path,primary_context))

        self.logger.info(f"共筛选出{len(target_img_list)}张需要处理的图片")
        return target_img_list

    # todo 明天继续
    def _find_image_contexts_in_md(self, MD_content, img_name, max_chars=100)->Tuple[str,str,str]:
        """
        基于MD文件的语义结构获取图片的前一个标题，上文，下文
        策略：
        1、将MD_content文件切分成以行为元素的列表
        2、根据MD文件格式，写出图片的正则表达式(含图片名称)
        3、根据图片的正则表达式，一行行匹配ME文件列表，找出图片所在位置
        4、以图片所在的位置为起点，找出离图片最近的上下标题
        5、截取图片与标题之间的内容作为上下文
        :param MD_content:
        :param img_name:
        :return: 返回一个含有图片前一个标题、上文、下文的列表
        """
        MD_content_lines_list=MD_content.split("\n")

        #图片的正则表达式
        import re
        image_pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)"
        )

        re_recently_heading=""
        re_recently_heading_index=-1
        final_img_pre_context=""
        final_img_next_context = ""

        self.log_step("step2.3.1","获取图片的位置")
        for img_index,line in enumerate(MD_content_lines_list):
            if not image_pattern.search(line):
                continue

            self.log_step("step2.3.2","获取图片的上一个标题位置及内容")
            for i in range(img_index-1,-1,-1):
                if re.match(r"^#{1,6}\s+",MD_content_lines_list[i]):
                    re_recently_heading=MD_content_lines_list[i].strip()
                    re_recently_heading_index=i
                    break

            self.log_step("step2.3.3", "获取图片的上文")
            pre_start=re_recently_heading_index+1 if re_recently_heading_index>=0 else 0
            img_pre_context=MD_content_lines_list[pre_start:img_index]
            final_img_pre_context=self._extract_paragraphs_with_limit(
                img_pre_context, max_chars, direction="backward"
            )

            self.log_step("step2.3.4", "获取图片的下一个标题位置")
            next_recently_heading_index=len(MD_content_lines_list)
            for i in range(img_index+1,next_recently_heading_index):
                if re.match(r"^#{1,6}\s+",MD_content_lines_list[i]):
                    next_recently_heading_index=i
                    break

            self.log_step("step2.3.5", "获取图片的下文")
            img_next_context=MD_content_lines_list[img_index+1,next_recently_heading_index]
            final_img_next_context=self._extract_paragraphs_with_limit(
                img_next_context,max_chars,direction="forward"
            )

        self.log_step("step2.3.6", "返回获取的图片上一个标题，上文，下文")
        return "".join(re_recently_heading),"".join(final_img_pre_context),"".join(final_img_next_context)

    def _extract_paragraphs_with_limit(self, pre_or_next_context, max_chars, direction):

        paragraphs = []
        current_para = []

        self.log_step("step2.3.5.1","去除空行和图片行")
        for line in pre_or_next_context:
            stripped = line.strip()
            if stripped == "":
                if current_para:
                    paragraphs.append("\n".join(current_para))
                    current_para = []
            else:
                # 跳过图片行
                if re.match(r"^!\[.*?\]\(.*?\)$", stripped):
                    if current_para:
                        paragraphs.append("\n".join(current_para))
                        current_para = []
                    continue
                current_para.append(stripped)

        if current_para:
            paragraphs.append("\n".join(current_para))

        paragraphs = [p for p in paragraphs if p.strip()]

        if not paragraphs:
            return ""

        if direction=="backward":
            paragraphs=list(reversed(paragraphs))

        self.log_step("step2.3.5.2", "在字数限制内截取上下文")
        selected_para=[]
        total_count=0

        for para in paragraphs:
            para_len = len(para)
            if total_count + para_len > max_chars and selected_para:
                break
            selected_para.append(para)
            total_count +=para_len

        if direction=="backward":
            selected_para=list(reversed(selected_para))

        return selected_para

    def get_img_summaries(self,target_img_list)->Dict[str,str]:
        """
        调用多模态模型为图片生成内容摘要
        :param target_img_list:待处理的图片信息列表
        :return:
        """
        img_summaries = {}

        self.log_step("step3.1","构建调用大模型的准备资料：llm客户端对象，模型，图片基础数据")
        #创建llm客户端对象
        try:
            client=OpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_api_base
            )
        except ImportError as e:
            self.logger.error(f"未安装openai库，无法初始化VL客户端,原因是{e}")
            return img_summaries
        except Exception as e:
            self.logger.error(f"初始化LV客户端失败，原因是{e}")
            return img_summaries

        model=self.config.vl_model

        self.log_step("step3.2", "构建调用大模型的方法")
        for img_name,img_full_path,content_tuple in target_img_list:
            single_img_sum=self.gene_sum_by_llm(
                client,
                model,
                img_full_path,
                content_tuple
            )
            img_summaries[img_name]=single_img_sum
        return img_summaries

    def gene_sum_by_llm(self, client, model, img_full_path,content_tuple):

        self.log_step("step3.2.1", "把图片转换成Base64编码的字符串，传给大模型")
        try:
            with open(img_full_path,"rb") as img_file:
                base64_image=base64.b64encode(img_file.read()).decode("utf-8")
        except IOError as e:
            self.logger.error(f"无法读取{img_full_path}图片文件:{e}")
            return "图片读取失败"

        self.log_step("step3.2.2", "构建提示词中图片的上下文")
        section_heading, pre_text, post_text=content_tuple
        wanshan_content_tuple=[]
        if section_heading:
            wanshan_content_tuple.append(f"图片所属章节标题：{section_heading}")
        if pre_text:
            wanshan_content_tuple.append(f"图片上文：{pre_text}")
        if post_text:
            wanshan_content_tuple.append(f"图片下午：{post_text}")

        content_info="\n".join(wanshan_content_tuple) if wanshan_content_tuple else "暂无可用上下文"

        self.log_step("step3.2.3", "整合提示词")
        messages=[
            {"role":"user",
             "content":[
                 {
                     "type":"text",
                     "text":f"任务：为MD文件中的图片生成一个简短的中文标题。"
                            f"背景信息：所属图片的原文档标题及上下文{content_info}"
                 },
                 {
                     "type":"image_url",
                     "image_url":{"url":f"data:image/jpeg;base64,{base64_image}"}
                 }
             ]
             }
        ]

        self.log_step("step3.2.4", "调用VL大模型")
        try:
            response=client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3
            )
            single_img_sum=response.choices[0].message.content.strip()
            return single_img_sum
        except LLMError as e:
            self.logger.error(f"调用大模型解析图片失败：{e}")

    def _upload_images_and_replaces_md_content(self, minio_client, target_img_list, img_summaries,MD_content):
        minio_imgName_imgUrl={}
        for img_name,img_full_path,_ in target_img_list:
            object_name=f"zhangguizhiku/{img_name}"
            ext = os.path.splitext(img_full_path)[1].lower()
            content_type = f"image/{ext[1:]}" if ext.startswith(".") else "application/octet-stream"

            if minio_client:

                try:
                    minio_client.fput_object(
                        bucket_name=self.config.minio_bucket,
                        object_name=object_name,
                        file_path=img_full_path,
                        content_type=content_type
                    )
                    minio_url=f"http://"+self.config.minio_endpoint+"/"+self.config.minio_bucket+"/"+object_name
                    minio_imgName_imgUrl[img_name]=minio_url
                    self.log_step("step4.1",f"图片上传成功,{img_name}->{minio_url}")
                except Exception as e:
                    self.logger.warning(f"图片{img_name}上传失败：{e}")

            else:
                self.logger.warning("客户端未初始化，跳过实际上传")

        new_md_content=MD_content
        for img_name,img_summary in img_summaries.items():
            minio_url=minio_imgName_imgUrl.get(img_name)
            if not minio_url:
                continue

            replace_pattern=re.compile(r"!\[(.*?)\]\((.*?" + re.escape(img_name) + r".*?)\)",re.IGNORECASE)
            new_md_content=replace_pattern.sub(f"![{img_summary}]({minio_url})",new_md_content)
        self.log_step("step4.2",f"成功替换MD文件中的{len(minio_imgName_imgUrl)}张图片")
        return new_md_content

    def _backup_new_md_file(self,original_path: Path,new_md_content: str) -> str:
        """
        将处理后的 Markdown 内容写入新文件。

        Args:
            original_md_path_str (str): 原始文件路径。
            new_md_content (str): 新的 Markdown 内容。

        Returns:
            str: 新文件的绝对路径。
        """
        self.log_step("step_5", "备份新文件")


        new_file_path = original_path.with_name(
            f"{original_path.stem}_new{original_path.suffix}"
        )

        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(f"文件写入失败: {e}", node_name=self.name)
        return str(new_file_path)



if __name__=="__main__":

    setup_logging()

    logger=logging.getLogger(__name__)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    state={
        "md_path":r"D:\$AAA\4、Large Models\项目\项目1\2.资料\2-pdf文档\pdf文档\doc\6W100-整本手册\auto\6W100-整本手册.md"
    }

    try:
        MD_Img_summ_Node=MDImgNode()
        MD_Img_summ_Node.process(state)
    except Exception as e:
        logger.exception(f"运行发生异常，异常原因是{e}")


