"""
约束解码器 —— 将图结构约束转化为生成引导信号

两种约束模式：
- 软约束（Soft Constraint）：将图结构转化为概率偏置，引导模型按路径生成
- 硬约束（Hard Constraint）：强制模型按拓扑顺序生成，不允许逻辑跳跃

实际使用建议以软约束为主，避免过度限制生成多样性
"""

from typing import Dict, List, Optional
from ..graph_representation.reasoning_graph import ReasoningGraph


class ConstrainedDecoder:
    """
    图约束解码器
    
    核心思想：
    将推理图的邻接依赖关系映射为LLM生成过程中的约束信号，
    强制模型按照规划路径进行推导，抑制非相关分支的干扰。
    
    实现方式：
    1. 基于执行计划的Prompt引导（主要方式）
    2. 注意力掩码约束（高级方式，需修改模型内部）
    3. 输出格式约束（JSON Schema约束输出结构）
    """

    def __init__(self, constraint_mode: str = "soft"):
        """
        Args:
            constraint_mode: 
                - "soft": 软约束（概率偏置/引导Prompt）—— 推荐
                - "hard": 硬约束（强制输出格式/掩码）
                - "hybrid": 混合约束
        """
        self.constraint_mode = constraint_mode

    def apply_constraint(self, 
                         graph: ReasoningGraph, 
                         execution_plan: Dict,
                         base_prompt: str) -> str:
        """
        将图约束应用到生成Prompt中
        
        Args:
            graph: 推理图
            execution_plan: 执行计划（来自PathPlanner）
            base_prompt: 基础推理Prompt
        
        Returns:
            添加了约束信号的Prompt
        """
        if self.constraint_mode == "soft":
            return self._soft_constraint(graph, execution_plan, base_prompt)
        elif self.constraint_mode == "hard":
            return self._hard_constraint(graph, execution_plan, base_prompt)
        else:
            return self._hybrid_constraint(graph, execution_plan, base_prompt)

    def _soft_constraint(self, graph: ReasoningGraph, plan: Dict, prompt: str) -> str:
        """
        软约束 —— 在Prompt中添加推理路径引导
        
        策略：将执行计划作为"思维框架"注入Prompt，
        引导模型按照规划路径思考，但不强制
        """
        # 格式化执行计划
        from .path_planner import PathPlanner
        planner = PathPlanner()
        plan_text = planner.format_execution_plan(graph, plan)

        constraint_prompt = f"""{prompt}

【推理路径规划】
请严格按照以下逻辑步骤进行推理，每个步骤的结论必须基于其前驱步骤的结果：

{plan_text}

【约束要求】
1. 必须按上述步骤顺序进行推理，不得跳步
2. 每一步推导必须引用前一步的结论
3. 如果某一步无法推导，请明确说明并尝试替代路径
4. 最终答案必须汇总所有关键步骤的结果

请开始推理："""

        return constraint_prompt

    def _hard_constraint(self, graph: ReasoningGraph, plan: Dict, prompt: str) -> str:
        """
        硬约束 —— 强制输出结构化JSON，要求逐步推理
        
        策略：要求模型按JSON Schema输出，
        每个推理步骤必须标注所依赖的前驱步骤
        """
        from .path_planner import PathPlanner
        planner = PathPlanner()
        plan_text = planner.format_execution_plan(graph, plan)

        constraint_prompt = f"""{prompt}

【强制推理约束】
你必须严格按照以下步骤顺序推理，并以JSON格式输出：

{plan_text}

请按以下JSON格式输出推理过程：
```json
{{
    "steps": [
        {{
            "step_id": 1,
            "content": "推理内容",
            "depends_on": [],
            "conclusion": "本步结论"
        }},
        {{
            "step_id": 2,
            "content": "推理内容",
            "depends_on": [1],
            "conclusion": "本步结论"
        }}
    ],
    "final_answer": "最终答案"
}}
```

要求：depends_on必须引用已完成步骤的step_id，不允许引用后续步骤。只输出JSON。"""

        return constraint_prompt

    def _hybrid_constraint(self, graph: ReasoningGraph, plan: Dict, prompt: str) -> str:
        """
        混合约束 —— 软约束为主体，关键节点处施加硬约束
        
        策略：
        - 全局：软约束引导路径
        - 分支点：硬约束要求明确选择
        - 汇聚点：硬约束要求引用所有前驱
        """
        soft = self._soft_constraint(graph, plan, prompt)
        
        # 在分支点和汇聚点添加额外约束
        branch_hints = []
        for bp_id in plan.get("branch_points", []):
            node = graph.get_node(bp_id)
            if node:
                neighbors = graph.get_neighbors(bp_id)
                branch_hints.append(
                    f"⚠️ 分支点「{node.content}」：需要分别探索{len(neighbors)}个方向"
                )
        
        merge_hints = []
        for mp_id in plan.get("merge_points", []):
            node = graph.get_node(mp_id)
            if node:
                preds = graph.get_predecessors(mp_id)
                merge_hints.append(
                    f"⚠️ 汇聚点「{node.content}」：必须汇总{len(preds)}个前驱步骤的结论"
                )

        if branch_hints or merge_hints:
            additional = "\n".join(branch_hints + merge_hints)
            soft += f"\n\n【关键节点约束】\n{additional}"

        return soft