import re
from typing import Dict, List, Optional, Tuple

from cryptography.x509 import name

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    # ========================================================================
    # 目录页处理 — 正则模式与参数
    # ========================================================================
    # 目录起始标记
    TOC_START_PATTERNS: List[re.Pattern] = [
        re.compile(r'^#{1,3}\s*(目\s*[录次]|Contents?|Table\s+of\s+Contents?)', re.IGNORECASE),
        re.compile(r'^\s*目\s*[录次]\s*$'),
    ]
    # 目录条目模式（章节编号 + 标题文字 + 页码）
    TOC_ENTRY_PATTERNS: List[re.Pattern] = [
        re.compile(r'^\s*(第[一二三四五六七八九十百\d]+章)\s*.+\d{1,4}\s*$'),
        re.compile(r'^\s*(\d+(?:\.\d+)+)\s+.+\d{1,4}\s*$'),
        re.compile(r'^\s*(\d+)\s+[^\d].*\d{1,4}\s*$'),
        re.compile(r'^\s*(附录\s*[A-Za-z]|Appendix\s+[A-Za-z])\s*.+\d{1,4}\s*$'),
    ]
    # 页码引导符：3个以上点号/省略号/空格 + 末尾数字
    PAGE_NUM_SEP = re.compile(r'[.…\s]{3,}\s*\d{1,4}\s*$')
    # 检测参数
    TOC_DENSITY_THRESHOLD: float = 0.2   # 目录条目在窗口中的最低占比
    TOC_WINDOW_SIZE: int = 20            # 检测窗口行数
    TOC_END_GAP: int = 3                 # 连续非目录行超过此数判定目录区域结束

    def process(self, state: ImportGraphState) -> ImportGraphState:

        #第一步：从state中导入文档
        self.log_step(f"{self.name},step1","获取文档内容")
        md_content,file_title=self._get_input_param(state)
        self.logger.info(f"导入md文档成功，返回{file_title}")

        # ---- 目录页处理：检测 → 解析 → 移除 ----
        self.log_step(f"{self.name},step1.5","检测并处理目录页")
        md_content, toc_structure = self._handle_toc(md_content)
        state["md_content"] = md_content
        state["toc_structure"] = toc_structure

        #第二步：将导入的文档按标题初次切分
        self.log_step(f"{self.name},step2", "将导入的文档按标题初次切分")
        initial_chunks=self._split_document_by_title(md_content,file_title)
        import json
        self.logger.info(f"文档初次切分成功，返回{json.dumps(initial_chunks, ensure_ascii=False, indent=2)}")

        #第三步：对初次切分好的内容进行二次切分，过大则二次切，过小则合并
        self.log_step(f"{self.name},step3","对初次切分好的内容进行二次切分，过大则二次切，过小则合并")
        final_chunks=self._split_and_merge(initial_chunks)
        """
        {
             "body":content,
             "title":new_title,
             "parent_title":parent_title,
             "file_title":file_title
            }
        """

        #第四步：将最终切分好的chunks更新到state里
        state["chunks"]=final_chunks
        self.logger.info(f"state['chunks']:{json.dumps(state['chunks'], ensure_ascii=False, indent=2)}")
        return state
        #第五步：（可选）输出操作日志和备份json格式


    def _split_and_merge(self,initial_chunks:list[Dict])->list[Dict]:
        """
          {
            "body": "88888888888888888888888888r",
            "title": "### 1.3 节点在流程中的位置",
            "parent_title": "## 1. 任务目标",
            "file_title": "6W100-整本手册"
          }
        :param initial_chunks:
        :return:
        """
        #3.1遍历初次切分的chunks列表
        collect_large_split_result=[]
        for initial_chunk in initial_chunks:
            if not initial_chunk.get("body"):
                continue
            total_len=len(initial_chunk.get("title"))+len(initial_chunk.get("body"))
            if total_len>self.config.max_content_length:
                # 3.2chunk过大则用递归字符文本切分器再次切分，得到large_split_result
                large_split_result=self._split_again(initial_chunk)
                # print(f"large_split_result:{large_split_result}")
                collect_large_split_result.extend(large_split_result)
            else:
                collect_large_split_result.append(initial_chunk)

        #3.3对collect_large_split_result中过小的部分进行合并，得到最终的final_split_result
        final_chunks=self.merge_short_part(collect_large_split_result)
        return final_chunks

    def _split_again(self, initial_chunk):
        """
            {
             "body":content,
             "title":new_title,
             "parent_title":parent_title,
             "file_title":file_title
            }
        :param initial_chunk:
        :return:
        """

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n","\n","。","！","？"],
            chunk_size=self.config.max_content_length-len(initial_chunk["title"]),
            chunk_overlap=0
        )
        texts=text_splitter.split_text(initial_chunk["body"])
        large_split_result = []
        for index,text in enumerate(texts):
            large_split_result.append(
                {
                    "body":text,
                    "title":initial_chunk["title"]+f"-{index+1}",
                    "parent_title":initial_chunk.get("parent_title"),
                    "chapter_path":initial_chunk.get("chapter_path",""),
                    "file_title":initial_chunk.get("file_title")
                }
            )
        return large_split_result

    def merge_short_part(self, collect_large_split_result):
        final_chunks=[]

        current_part = collect_large_split_result[0]
        # print(f"current_part:{current_part}")

        for part in collect_large_split_result[1:]:
            len_current_part=len(current_part["body"])+len(current_part["title"])
            len_part=len(part["body"])+len(part["title"])

            same_parent_title=current_part.get("parent_title")==part.get("parent_title")
            #同源且当前块小于最小限制
            if same_parent_title and len_current_part<self.config.min_content_length:
                #合并前判断合并后是否大于最大限制
                if len_current_part+len_part>=self.config.max_content_length:
                    final_chunks.append(current_part)
                    current_part=part
                    continue
                current_part["body"]=current_part.get("body").rstrip()+"\n\n"+part.get("body").lstrip()
                # current_part["title"]=current_part.get("parent_title")
                continue

            else:
                final_chunks.append(current_part)
                current_part=part
        final_chunks.append(current_part)
        return final_chunks

    def _split_document_by_title(self,md_content,file_title):
        """
        根据标题切分，遇到标题结束，切分上一段
        :param md_content:
        :param file_title:
        :return:
        """
        self.log_step("step2.1","将md文档拆分成列表")
        #将md文本按换行符拆分成列表
        lines=md_content.split("\n")

        self.log_step("step2.2","文档切分前的准备工作，准备各种列表、字符串来装东西")
        #创建查到标题的正则表达式
        title_rule=re.compile(r"^\s*(#{1,6})\s+(.+)")

        #临时储存遍历到的列
        tem=[]

        # 是否代码块
        in_code = False

        # 当前文档的上一个标题
        new_title = ""

        #标题级别列表
        level_title_list=[""]*7
        current_level=0

        #resp最终数据
        resp=[]

        self.log_step("step2.3","开始一行行遍历md_content列表")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code=not in_code

            self.log_step("step2.3.1", "如果是标题，就保存本次标题供下一轮文档使用，并计算上一次标题的父标题")
            #如果是标题且不是在代码块内
            if not in_code and title_rule.search(line):
                #将临时储存的列表行赋值给content
                content="\n".join(tem)
                #如果content不为空，把前面临时储存的内容合并，并找到他的父标题
                if new_title or content:
                    parent_title=""
                    #从当前标题级别往上找
                    for lv in range(current_level-1,0,-1):
                        #如果标题级别列表不为空，说明目前标题往上级别找到了一个标题，这个标题作为本次content的父标题
                        if level_title_list[lv]:
                            parent_title=level_title_list[lv]
                            break
                    #如果往上遍历完了，都没有标题，就默认一个
                    if not parent_title:
                        parent_title=new_title if new_title else file_title

                    chapter_path = self._build_chapter_path(level_title_list, current_level)
                    resp.append(
                        {"body":content,
                         "title":new_title,
                         "parent_title":parent_title,
                         "chapter_path":chapter_path,
                         "file_title":file_title
                         }
                    )


                self.log_step("step2.3.1.2", "将当前标题存到标题级别列表中")
                current_title_obj=title_rule.match(line)
                if current_title_obj:
                    level=len(current_title_obj.group(1))
                    current_level=level
                    level_title_list[level]=line

                    # 当前列表存储标题索引 后面索引值清空，避免层次混乱
                    for lv in range(current_level + 1, 7):
                        level_title_list[lv] = ""

                new_title=line
                tem=[]

            else:
                self.log_step("step2.3.2", "如果不是标题，就把内容加到临时列表中")
                tem.append(line)

        self.log_step("step2.3.3","最后一段内容后面没有标题，需要单独加入")
        last_content="\n".join(tem)
        if new_title or last_content:
            parent_title=""
            for lv in range(current_level-1,0,-1):
                if level_title_list[lv]:
                    parent_title=level_title_list[lv]
                    break
            if not parent_title:
                parent_title=new_title if new_title else file_title

            chapter_path = self._build_chapter_path(level_title_list, current_level)
            resp.append(
                {"body": last_content,
                 "title": new_title,
                 "parent_title": parent_title,
                 "chapter_path": chapter_path,
                 "file_title": file_title
                 }
            )

        return resp

    def _get_input_param(self, state):
        """
        获取md文档内容
        :param state:
        :return:
        """
        md_content=state.get("md_content")
        if md_content:
            md_content=(md_content.replace("\r\n","\n")).replace("\r","\n")

        file_title=state.get("file_title")
        return md_content,file_title

    # ========================================================================
    # 目录页处理
    # ========================================================================

    def _handle_toc(self, md_content: str) -> Tuple[str, List[Dict]]:
        """
        目录页处理主方法：检测 → 解析 → 移除。

        算法概述：
          1. 按行扫描，找到匹配 TOC_START_PATTERNS 的行作为目录区域起点
          2. 从起点后取窗口行，计算目录条目密度
          3. 密度达标 → 确认是目录页，向后扩展直到连续 N 行非目录条目
          4. 从目录行列表中解析树形章节结构
          5. 从 md_content 中切除目录区域

        Returns:
            (清洗后的 md_content, toc_structure 列表)
        """
        lines = md_content.split("\n")
        original_count = len(lines)

        # 1. 定位目录区域
        toc_range = self._detect_toc_region(lines)
        if toc_range is None:
            self.logger.info("未检测到目录页，跳过")
            return md_content, []

        start_line, end_line = toc_range
        self.logger.info(
            f"检测到目录页: 第 {start_line + 1} → {end_line + 1} 行 "
            f"（共 {end_line - start_line + 1} 行）"
        )

        # 2. 解析目录树形结构
        toc_lines = lines[start_line:end_line + 1]
        toc_structure = self._parse_toc_structure(toc_lines)
        self.logger.info(f"解析出 {len(toc_structure)} 条目录结构")

        # 3. 从 md_content 中移除目录区域
        remaining = lines[:start_line] + lines[end_line + 1:]
        cleaned = "\n".join(remaining)
        self.logger.info(
            f"已移除目录页: {original_count} 行 → {len(remaining)} 行"
        )

        return cleaned, toc_structure

    def _detect_toc_region(self, lines: List[str]) -> Optional[Tuple[int, int]]:
        """
        检测目录页区域的起止行号。

        三重判定（降低误判）：
          特征1 — 标题行包含"目录/目次/Contents"
          特征2 — 窗口内目录条目密度 ≥ TOC_DENSITY_THRESHOLD
          特征3 — 遇到明确的新章节标题或连续空行 ≥ TOC_END_GAP 即结束
        """
        # 1. 找目录起始行
        toc_start = None
        for i, line in enumerate(lines):
            for pat in self.TOC_START_PATTERNS:
                if pat.search(line):
                    toc_start = i
                    break
            if toc_start is not None:
                break

        if toc_start is None:
            return None

        # 2. 密度验证
        window_size = min(self.TOC_WINDOW_SIZE, len(lines) - toc_start - 1)
        if window_size < 3:
            return None

        check_start = toc_start + 1
        check_end = min(toc_start + 1 + window_size, len(lines))
        entry_count = sum(
            1 for i in range(check_start, check_end)
            if self._is_toc_entry(lines[i])
        )
        density = entry_count / max(window_size, 1)

        if density < self.TOC_DENSITY_THRESHOLD:
            self.logger.info(
                f"目录条目密度过低（{density:.0%} < {self.TOC_DENSITY_THRESHOLD:.0%}），跳过"
            )
            return None

        # 3. 找目录结束行
        toc_end = check_start
        consecutive_non_toc = 0

        for i in range(check_start, len(lines)):
            if self._is_toc_entry(lines[i]):
                consecutive_non_toc = 0
                toc_end = i
                continue

            stripped = lines[i].strip()

            # 遇到非目录类标题 → 结束
            is_heading = re.match(r'^#{1,3}\s+', stripped)
            is_toc_heading = any(pat.search(stripped) for pat in self.TOC_START_PATTERNS)
            if is_heading and not is_toc_heading:
                break

            consecutive_non_toc += 1
            if consecutive_non_toc >= self.TOC_END_GAP:
                break

            toc_end = i

        if toc_end <= toc_start + 1:
            return None

        return (toc_start, toc_end)

    def _is_toc_entry(self, line: str) -> bool:
        """
        判断一行是否为目录条目。

        排除：空行、markdown 标题行、纯数字行。
        """
        stripped = line.strip()
        if not stripped:
            return False
        if re.match(r'^#{1,6}\s+', stripped):
            return False
        if re.match(r'^\d{1,4}\s*$', stripped):
            return False

        for pat in self.TOC_ENTRY_PATTERNS:
            if pat.search(stripped):
                return True
        if self.PAGE_NUM_SEP.search(stripped):
            return True

        return False

    def _parse_toc_structure(self, toc_lines: List[str]) -> List[Dict]:
        """
        从目录行列表解析树形章节结构。

        Returns:
            [
                {"level": 1, "number": "第一章", "title": "产品概述", "page": 1},
                {"level": 2, "number": "1.1",   "title": "产品简介", "page": 3},
                ...
            ]
        """
        structure = []
        for line in toc_lines:
            entry = self._parse_single_toc_line(line)
            if entry:
                structure.append(entry)
        return structure

    def _parse_single_toc_line(self, line: str) -> Optional[Dict]:
        """解析单行目录条目，提取 层级/编号/标题/页码"""
        stripped = line.strip()
        if not stripped or not self._is_toc_entry(stripped):
            return None

        # 提取页码（行尾数字）
        page_num = None
        pm = re.search(r'[.…\s]{2,}\s*(\d{1,4})\s*$', stripped)
        if pm:
            page_num = int(pm.group(1))
            stripped = re.sub(r'[.…\s]{2,}\s*\d{1,4}\s*$', '', stripped).strip()
        else:
            pm = re.search(r'\s+(\d{1,4})\s*$', stripped)
            if pm:
                page_num = int(pm.group(1))
                stripped = re.sub(r'\s+\d{1,4}\s*$', '', stripped).strip()

        # 提取层级和编号
        level = 1
        number = ""
        title = stripped

        # "第一章 xxx"
        ch_match = re.match(r'^(第[一二三四五六七八九十百\d]+章)\s*(.*)', stripped)
        if ch_match:
            number = ch_match.group(1)
            title = ch_match.group(2) or stripped
            level = 1
        # "1.1.1 xxx" 或 "1.1 xxx"
        elif re.match(r'^\d+(?:\.\d+)+\s+', stripped):
            num_match = re.match(r'^(\d+(?:\.\d+)+)\s+(.+)', stripped)
            if num_match:
                number = num_match.group(1)
                title = num_match.group(2)
                level = number.count(".") + 1
        # "1  xxx"
        elif re.match(r'^(\d+)\s+([^\d].*)', stripped):
            num_match = re.match(r'^(\d+)\s+([^\d].*)', stripped)
            if num_match:
                number = num_match.group(1)
                title = num_match.group(2)
                level = 1
        # "附录A xxx" / "Appendix A xxx"
        elif re.match(r'^(附录\s*[A-Za-z]|Appendix\s+[A-Za-z])\s*(.*)', stripped):
            app_match = re.match(r'^(附录\s*[A-Za-z]|Appendix\s+[A-Za-z])\s*(.*)', stripped)
            if app_match:
                number = app_match.group(1)
                title = app_match.group(2) or stripped
                level = 1

        return {
            "level": level,
            "number": number,
            "title": title.strip(),
            "page": page_num,
        }

    @staticmethod
    def _build_chapter_path(level_title_list: list, current_level: int) -> str:
        """
        从标题栈构建完整章节路径。

        在切分时调用，将当前激活的标题栈拼接为层级路径，
        供 embedding 文本和 Milvus output_field 使用。

        Args:
            level_title_list: 标题级别列表，索引=级别（1-6），值=标题文本（含 # 标记）
            current_level: 当前标题级别

        Returns:
            章节路径，如 "第一章 产品概述 > 1.1 产品简介 > 1.1.1 功能概述"
        """
        parts = []
        for lv in range(1, min(current_level + 1, len(level_title_list))):
            if level_title_list[lv]:
                clean = re.sub(r'^#{1,6}\s*', '', level_title_list[lv]).strip()
                if clean:
                    parts.append(clean)
        return " > ".join(parts) if parts else ""




if __name__ == '__main__':
    setup_logging()
    doc_split_node = DocumentSplitNode()
    # 构建md_content
    file_path = r"D:\$AAA\4、Large Models\项目\test.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    # 构建数据
    state = {
        "file_title": "6W100-整本手册",
        "md_content": file_content,
    }

    doc_split_node.process(state)

