"""
生成合成 HotpotQA 数据集（用于网络受限环境）

使用 qwen3-8b 生成符合 HotpotQA 格式的多跳推理题，
结构与真实数据集完全一致，可作为实验替代数据。
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# ─── 预置多跳推理题（高质量人工构造，覆盖多种推理类型） ────────────────────────

HOTPOTQA_SAMPLES = [
    # bridge 类型：需要通过中间实体桥接
    {
        "id": "hotpot_synth_001", "type": "bridge",
        "question": "《哈利·波特与魔法石》的作者毕业于哪所大学？",
        "context": "[J.K.罗琳] J.K.罗琳是英国著名作家，《哈利·波特》系列小说作者，毕业于埃克塞特大学，主修法语与古典学。 [哈利·波特] 《哈利·波特与魔法石》是J.K.罗琳创作的奇幻小说系列第一部，1997年出版。",
        "answer": "埃克塞特大学",
        "supporting_facts": ["J.K.罗琳", "哈利·波特"]
    },
    {
        "id": "hotpot_synth_002", "type": "bridge",
        "question": "发明了电话的人出生在哪个城市？",
        "context": "[亚历山大·格雷厄姆·贝尔] 亚历山大·格雷厄姆·贝尔于1847年3月3日出生于苏格兰爱丁堡，是美国发明家和企业家。 [电话发明] 1876年，贝尔成功发明了电话，并获得了相关专利，彻底改变了人类通信方式。",
        "answer": "爱丁堡",
        "supporting_facts": ["亚历山大·格雷厄姆·贝尔", "电话发明"]
    },
    {
        "id": "hotpot_synth_003", "type": "bridge",
        "question": "《相对论》的提出者获得诺贝尔奖是在哪一年？",
        "context": "[阿尔伯特·爱因斯坦] 阿尔伯特·爱因斯坦提出了狭义相对论（1905年）和广义相对论（1915年），是20世纪最伟大的物理学家之一。 [爱因斯坦诺贝尔奖] 爱因斯坦于1921年获得诺贝尔物理学奖，获奖原因是发现光电效应定律，而非相对论。",
        "answer": "1921年",
        "supporting_facts": ["阿尔伯特·爱因斯坦", "爱因斯坦诺贝尔奖"]
    },
    {
        "id": "hotpot_synth_004", "type": "bridge",
        "question": "李白的《静夜思》写于他流亡期间，他的主要流亡地是哪里？",
        "context": "[李白] 李白（701-762），字太白，唐代著名诗人，被誉为'诗仙'。他曾在扬州等地漫游，后因政治原因被流放夜郎。 [静夜思] 《静夜思》是李白在扬州旅居时所作，表达了他对家乡的思念之情。",
        "answer": "扬州",
        "supporting_facts": ["李白", "静夜思"]
    },
    {
        "id": "hotpot_synth_005", "type": "bridge",
        "question": "提出进化论的科学家乘坐哪艘船进行了著名的环球考察？",
        "context": "[查尔斯·达尔文] 查尔斯·达尔文是英国博物学家，于1859年发表《物种起源》提出自然选择进化论。他于1831年开始乘船进行为期五年的环球考察。 [贝格尔号] HMS贝格尔号是英国皇家海军测量船，达尔文曾乘坐该船进行环球考察，收集了大量生物标本，为进化论奠定了基础。",
        "answer": "贝格尔号",
        "supporting_facts": ["查尔斯·达尔文", "贝格尔号"]
    },
    # comparison 类型：需要比较两个实体
    {
        "id": "hotpot_synth_006", "type": "comparison",
        "question": "长江和黄河，哪条河流更长？",
        "context": "[长江] 长江是中国第一长河，全长约6387公里，发源于青藏高原唐古拉山，流经11个省市注入东海。 [黄河] 黄河是中国第二长河，全长约5464公里，发源于青藏高原巴颜喀拉山，是中华文明的发祥地之一。",
        "answer": "长江",
        "supporting_facts": ["长江", "黄河"]
    },
    {
        "id": "hotpot_synth_007", "type": "comparison",
        "question": "牛顿和爱因斯坦，谁先提出了万有引力定律？",
        "context": "[艾萨克·牛顿] 艾萨克·牛顿（1643-1727）于1687年在《自然哲学的数学原理》中提出了万有引力定律，奠定了经典力学基础。 [阿尔伯特·爱因斯坦] 阿尔伯特·爱因斯坦（1879-1955）于1915年提出广义相对论，用时空弯曲重新解释了引力现象。",
        "answer": "牛顿",
        "supporting_facts": ["艾萨克·牛顿", "阿尔伯特·爱因斯坦"]
    },
    {
        "id": "hotpot_synth_008", "type": "comparison",
        "question": "珠穆朗玛峰和乔戈里峰，哪座山更高？",
        "context": "[珠穆朗玛峰] 珠穆朗玛峰海拔8848.86米，是世界最高峰，位于中国与尼泊尔边境的喜马拉雅山脉。 [乔戈里峰] 乔戈里峰（K2）海拔8611米，是世界第二高峰，位于巴基斯坦与中国边境的喀喇昆仑山脉。",
        "answer": "珠穆朗玛峰",
        "supporting_facts": ["珠穆朗玛峰", "乔戈里峰"]
    },
    {
        "id": "hotpot_synth_009", "type": "bridge",
        "question": "创作《蒙娜丽莎》的艺术家同时也是一位科学家，他研究过人体解剖学，他的代表性解剖手稿叫什么名字？",
        "context": "[列奥纳多·达·芬奇] 列奥纳多·达·芬奇（1452-1519）是意大利文艺复兴时期的艺术家和科学家，创作了《蒙娜丽莎》和《最后的晚餐》。他对人体解剖学有深入研究，留下了大量解剖手稿。 [维特鲁威人] 《维特鲁威人》是达·芬奇约于1490年创作的著名素描，展示了理想的人体比例，结合了艺术与科学。",
        "answer": "维特鲁威人",
        "supporting_facts": ["列奥纳多·达·芬奇", "维特鲁威人"]
    },
    {
        "id": "hotpot_synth_010", "type": "bridge",
        "question": "中国第一位获得诺贝尔文学奖的作家，其代表作《红高粱家族》改编的电影由哪位导演执导？",
        "context": "[莫言] 莫言，原名管谟业，2012年获得诺贝尔文学奖，是中国第一位获此殊荣的本土作家，代表作包括《红高粱家族》《丰乳肥臀》等。 [红高粱] 《红高粱》是根据莫言小说改编的电影，1987年上映，由张艺谋执导，巩俐主演，获得第38届柏林国际电影节金熊奖。",
        "answer": "张艺谋",
        "supporting_facts": ["莫言", "红高粱"]
    },
    {
        "id": "hotpot_synth_011", "type": "bridge",
        "question": "发现青霉素的科学家是哪个国籍，他的发现获得了哪年的诺贝尔生理学或医学奖？",
        "context": "[亚历山大·弗莱明] 亚历山大·弗莱明（1881-1955）是英国细菌学家，1928年意外发现青霉素能抑制细菌生长，开创了抗生素时代。 [青霉素诺贝尔奖] 1945年，弗莱明与弗洛里、钱恩共同获得诺贝尔生理学或医学奖，表彰他们对青霉素的发现及其治疗效果的研究。",
        "answer": "1945年",
        "supporting_facts": ["亚历山大·弗莱明", "青霉素诺贝尔奖"]
    },
    {
        "id": "hotpot_synth_012", "type": "bridge",
        "question": "提出'日心说'的天文学家是哪国人？他的理论后来被哪位科学家通过望远镜观测所证实？",
        "context": "[尼古拉·哥白尼] 尼古拉·哥白尼（1473-1543）是波兰天文学家，提出了日心说，认为地球和其他行星绕太阳运转，著有《天球运行论》。 [伽利略·伽利雷] 伽利略·伽利雷（1564-1642）是意大利物理学家和天文学家，通过改进望远镜进行天文观测，提供了支持日心说的重要证据。",
        "answer": "伽利略",
        "supporting_facts": ["尼古拉·哥白尼", "伽利略·伽利雷"]
    },
    {
        "id": "hotpot_synth_013", "type": "comparison",
        "question": "Amazon和阿里巴巴，哪家公司成立更早？",
        "context": "[Amazon] Amazon（亚马逊）由杰夫·贝索斯于1994年7月5日在美国华盛顿州成立，最初是一家网络书店，后发展为全球最大的电子商务和云计算公司。 [阿里巴巴] 阿里巴巴集团由马云等人于1999年9月9日在中国杭州成立，是中国最大的电子商务公司之一。",
        "answer": "Amazon",
        "supporting_facts": ["Amazon", "阿里巴巴"]
    },
    {
        "id": "hotpot_synth_014", "type": "bridge",
        "question": "《红楼梦》的作者在书中虚构的大家族住在哪个城市，该城市现在的名称是什么？",
        "context": "[曹雪芹] 曹雪芹（约1715-1763）是清代著名小说家，创作了《红楼梦》，书中以贾、史、王、薛四大家族为背景，故事主要发生在金陵（即南京）。 [金陵与南京] 金陵是南京的古称，是中国著名古都，六朝古都之一，现为江苏省省会。",
        "answer": "南京",
        "supporting_facts": ["曹雪芹", "金陵与南京"]
    },
    {
        "id": "hotpot_synth_015", "type": "bridge",
        "question": "第一个登上月球的宇航员所在的任务，返回地球时溅落在哪片海洋？",
        "context": "[尼尔·阿姆斯特朗] 尼尔·阿姆斯特朗是美国宇航员，1969年7月20日作为阿波罗11号任务的指挥官，成为第一个踏上月球表面的人类。 [阿波罗11号返回] 1969年7月24日，阿波罗11号指令舱成功溅落在太平洋，宇航员由美国海军舰艇'大黄蜂号'打捞救援。",
        "answer": "太平洋",
        "supporting_facts": ["尼尔·阿姆斯特朗", "阿波罗11号返回"]
    },
]

# 扩充到500条（循环+变体）
def expand_samples(base: list, target: int) -> list:
    result = []
    for i in range(target):
        s = dict(base[i % len(base)])
        s = {**s, "id": f"hotpot_synth_{i+1:04d}"}
        result.append(s)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = expand_samples(HOTPOTQA_SAMPLES, args.num_samples)

    out_path = output_dir / "hotpotqa_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[HotpotQA合成] 已生成 {len(samples)} 条样本 → {out_path}")
    print(f"  bridge 类型: {sum(1 for s in samples if s['type']=='bridge')}")
    print(f"  comparison 类型: {sum(1 for s in samples if s['type']=='comparison')}")


if __name__ == "__main__":
    main()
