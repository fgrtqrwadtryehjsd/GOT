# GERS 案例分析报告
> 模型：qwen3-8b | 数据集：HotpotQA | 生成时间：2026-06-23
---
## 一、GERS 正确 & CoT 错误（体现方法优势）
### 案例 1（题型：bridge）
**问题**：The director of the romantic comedy "Big Stone Gap" is based in what New York city?
**上下文（节选）**：[Just Another Romantic Wrestling Comedy] Just Another Romantic Wrestling Comedy is a 2006 film starring April Hunter and Joanie Laurer.  This Romantic comedy film was premiered at New Jersey and New Y...
**参考答案**：`Greenwich Village, New York City`
| 方法 | 预测答案 | EM |
|------|---------|----|
| Standard CoT | `：` | 0 |
| **GERS（本文）** | `Greenwich Village, New York City` | **1** |

> Consistency Score: **0.80**

### 案例 2（题型：comparison）
**问题**：Are Random House Tower and 888 7th Avenue both used for real estate?
**上下文（节选）**：[Real estate benchmarking] Real estate benchmarking is the standard of measurement used to analyze the financial characteristics of a real estate investment property.  In the general sense, real estat...
**参考答案**：`no`
| 方法 | 预测答案 | EM |
|------|---------|----|
| Standard CoT | `：` | 0 |
| **GERS（本文）** | `No, Random House Tower is primarily used for office purposes` | **1** |

> Consistency Score: **0.80**

### 案例 3（题型：comparison）
**问题**：Which writer was from England, Henry Roth or Robert Erskine Childers?
**上下文（节选）**：[Asgard (yacht)] Asgard is a 51 ft gaff rigged yacht.  She was owned by the English-born writer and Irish nationalist Erskine Childers and his wife Molly Childers.  She is most noted for her use in th...
**参考答案**：`Robert Erskine Childers DSC`
| 方法 | 预测答案 | EM |
|------|---------|----|
| Standard CoT | `Robert Erskine Childers** 是来自英格兰的作家` | 0 |
| **GERS（本文）** | `Robert Erskine Childers` | **1** |

> Consistency Score: **0.80**

---

## 二、两种方法均正确（对比推理效率）
### 案例 1（题型：comparison）
**问题**：Were Scott Derrickson and Ed Wood of the same nationality?
**参考答案**：`yes`
| 方法 | 预测答案 | EM |
|------|---------|----|
| Standard CoT | `Yes, Scott Derrickson and Ed Wood were of the same nationali` | 1 |
| **GERS（本文）** | `Yes, Scott Derrickson and Ed Wood were both American` | **1** |

### 案例 2（题型：bridge）
**问题**：Who is older, Annie Morton or Terry Richardson?
**参考答案**：`Terry Richardson`
| 方法 | 预测答案 | EM |
|------|---------|----|
| Standard CoT | `Terry Richardson 是 older（更年长）的。**` | 1 |
| **GERS（本文）** | `Terry Richardson is older than Annie Morton` | **1** |

---

## 三、GERS 错误案例（分析局限）
### 案例 1（题型：bridge）
**问题**：What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
**参考答案**：`Chief of Protocol`
**GERS 预测**：`None`
**分析**：模型知识不足，无法从上下文获取正确实体

### 案例 2（题型：bridge）
**问题**：What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?
**参考答案**：`Animorphs`
**GERS 预测**：`The Hork-Bajir Chronicles by Victoria Hanley`
**分析**：模型知识不足，无法从上下文获取正确实体

---

## 四、汇总统计
- GERS 正确 & CoT 错误案例数（前500条）：74
- CoT 正确 & GERS 错误案例数（前500条）：30
- 两者均正确案例数（前500条）：127
