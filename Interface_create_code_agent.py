# # 根据已有的代码，根据用户提供的api，生成对应的代码
# import os
#
# import docx
# from docx import Document
# from langchain_core.tools import tool
#
# from DB.chroma import ChromaDB
# from common.modeCommon import Model
# from config.config import Config
# from enums.model_enums import ModelType
# from knowledge import Knowledge
#
# import logging
#
# logger = logging.getLogger(__name__)
# def load_file(file_path: list[str], document_processor: Knowledge, chromadb: ChromaDB):
#     logger.info("开始加载文件")
#     for file_path in file_path:
#         data = document_processor.load_files([file_path], "")
#         chromadb.add(data)
#         logger.info(f"加载文件{file_path}成功")
#
#
#
# @tool
# def read_docx(file_path):
#
#     """读取.docx文件内容（不包括页眉页脚）"""
#     if not os.path.exists(file_path):
#         print("文件不存在！")
#         return ""
#     doc = Document(file_path)
#     full_text = []
#
#     # 读取段落文本
#     for para in doc.paragraphs:
#         full_text.append(para.text)
#
#     # 读取表格内容
#     for table in doc.tables:
#         for row in table.rows:
#             for cell in row.cells:
#                 full_text.append(cell.text)
#
#     return "\n".join(full_text)
#
# def parse_api_info(model: Model, api_info: str):
#     """
#     解析api信息
#     :param model: 模型
#     :param api_info: api信息
#     :return: 解析后的api信息 api_name, api_url, api_method, api_params, api_desc
#     """
#     prompt = f"""
#     你是一个api信息解析器，你需要解析以下api信息，提取api_name, api_url, api_method, api_params, api_desc
#     api_info: {api_info}
#     """
#     response = model.qwen_llm().invoke(prompt)
#     return response.content
#
# def parse_api_info_agent(model: Model, api_info: str):
#     return parse_api_info(model, api_info)
#
#
#
#
# if __name__ == '__main__':
#     # file_path = [
#     #     'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\CodePeckerSASTManage.java',
#     #     'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\CodePeckerSASTPlugin.java',
#     #     'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\CodePeckerSASTService.java',
#     #     'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\CodeVulnInfo.java',
#     #     'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\PagedResponse.java',
#     #     'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\CodeVulnInfo.java',
#     #     "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\CommentReq.java",
#     #     "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\CreateProjectReq.java",
#     #     "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DetectionInfo.java",
#     #     "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DetectionReq.java",
#     #     "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DetectionRes.java",
#     #     "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DetectionStatusRes.java",
#     #     "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DownloadReq.java",
#     #     "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DownloadReq.java"]
#     # document_processor = Knowledge()
#     config = Config('conf/config.yml')
#     # chromadb = ChromaDB(config, ModelType.QWEN)
#     # load_file(file_path, document_processor,chromadb)
#     model = Model(config)
#     api_info = read_docx(r"D:\api.docx")
#     if api_info:
#         print(f"原始api信息：{api_info}")
#         print("="*30)
#         parse_api_info = parse_api_info_agent(model, api_info)
#
#         print(f"解析后的api信息：{parse_api_info}")
#     else:
#         print("文件不存在！")
#
#     print(parse_api_info)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# 设置页面
st.set_page_config(
    page_title="系统监控评分仪表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题和说明
st.title("📊 系统监控评分仪表板")
st.markdown("""
该仪表板展示系统各项监控指标的实时状态和综合评分，帮助您了解系统健康状态。
""")


# 模拟监控数据获取函数
def fetch_monitor_data():
    """模拟获取监控数据"""
    # 在实际应用中，这里会调用真实的API
    current_time = datetime.now()

    # 生成模拟数据
    data = {
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": {
            "cpu_usage": np.random.uniform(10, 90),
            "memory_usage": np.random.uniform(20, 85),
            "disk_usage": np.random.uniform(30, 80)
        },
        "mysql": {
            "connections": np.random.randint(50, 200),
            "queries": np.random.randint(1000, 5000),
            "slow_queries": np.random.randint(0, 15)
        },
        "redis": {
            "hit_rate": np.random.uniform(85, 99.9),
            "connections": np.random.randint(10, 100),
            "memory_used": np.random.uniform(500, 2000)  # MB
        },
        "services": {
            "api_service": np.random.choice([0, 1, 2], p=[0.85, 0.1, 0.05]),
            "auth_service": np.random.choice([0, 1, 2], p=[0.9, 0.05, 0.05]),
            "database_service": np.random.choice([0, 1, 2], p=[0.95, 0.03, 0.02]),
            "cache_service": np.random.choice([0, 1, 2], p=[0.88, 0.08, 0.04])
        },
        "es": {
            "cpu_usage": np.random.uniform(15, 75),
            "memory_usage": np.random.uniform(25, 80),
            "disk_usage": np.random.uniform(40, 90)
        }
    }
    return data


# 计算综合评分
def calculate_score(data):
    """根据监控数据计算综合评分"""
    scores = {}

    # 主机指标评分 (权重: 30%)
    host_score = 100 - (
            max(0, data["host"]["cpu_usage"] - 70) * 0.5 +
            max(0, data["host"]["memory_usage"] - 75) * 0.3 +
            max(0, data["host"]["disk_usage"] - 80) * 0.2
    )
    scores["host"] = max(60, min(100, host_score))

    # MySQL指标评分 (权重: 25%)
    mysql_score = 100 - (
            max(0, data["mysql"]["connections"] - 150) * 0.1 +
            data["mysql"]["slow_queries"] * 2
    )
    scores["mysql"] = max(60, min(100, mysql_score))

    # Redis指标评分 (权重: 20%)
    redis_score = 100 - (
            (100 - data["redis"]["hit_rate"]) * 0.5 +
            max(0, data["redis"]["connections"] - 80) * 0.2 +
            max(0, data["redis"]["memory_used"] - 1500) * 0.01
    )
    scores["redis"] = max(60, min(100, redis_score))

    # 服务状态评分 (权重: 15%)
    service_status = data["services"]
    service_penalty = (
            (1 if service_status["api_service"] == 1 else 2 if service_status["api_service"] == 2 else 0) +
            (1 if service_status["auth_service"] == 1 else 2 if service_status["auth_service"] == 2 else 0) +
            (1 if service_status["database_service"] == 1 else 2 if service_status["database_service"] == 2 else 0) +
            (1 if service_status["cache_service"] == 1 else 2 if service_status["cache_service"] == 2 else 0)
    )
    service_score = 100 - service_penalty * 10
    scores["services"] = max(60, min(100, service_score))

    # ES指标评分 (权重: 10%)
    es_score = 100 - (
            max(0, data["es"]["cpu_usage"] - 70) * 0.4 +
            max(0, data["es"]["memory_usage"] - 75) * 0.3 +
            max(0, data["es"]["disk_usage"] - 85) * 0.3
    )
    scores["es"] = max(60, min(100, es_score))

    # 综合评分
    weights = {
        "host": 0.3,
        "mysql": 0.25,
        "redis": 0.2,
        "services": 0.15,
        "es": 0.1
    }

    total_score = sum(scores[category] * weights[category] for category in scores)
    scores["total"] = total_score

    return scores


# 状态指示器
def status_indicator(value, thresholds=[80, 90]):
    """生成状态指示器"""
    if value < thresholds[0]:
        return st.success("正常")
    elif value < thresholds[1]:
        return st.warning("警告")
    else:
        return st.error("异常")


# 生成历史数据（模拟）
@st.cache_data(ttl=300)
def generate_historical_data(hours=24):
    """生成历史监控数据"""
    history = []
    current_time = datetime.now()

    for i in range(hours):
        timestamp = current_time - timedelta(hours=i)
        data = fetch_monitor_data()
        scores = calculate_score(data)

        history.append({
            "timestamp": timestamp,
            "scores": scores,
            "data": data
        })

    return history


# 侧边栏
st.sidebar.header("配置选项")
auto_refresh = st.sidebar.checkbox("自动刷新", value=True)
refresh_interval = st.sidebar.slider("刷新间隔(秒)", min_value=5, max_value=60, value=15)

if auto_refresh:
    st.sidebar.write(f"下次刷新: {refresh_interval}秒")
    time.sleep(refresh_interval)
    st.experimental_rerun()

# 获取数据
data = fetch_monitor_data()
scores = calculate_score(data)
history = generate_historical_data()

# 总体评分卡片
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("综合评分", f"{scores['total']:.1f}", delta="-2.5" if scores['total'] < 90 else "+1.2")

with col2:
    st.metric("主机评分", f"{scores['host']:.1f}")

with col3:
    st.metric("MySQL评分", f"{scores['mysql']:.1f}")

with col4:
    st.metric("Redis评分", f"{scores['redis']:.1f}")

# 评分趋势图
st.subheader("评分趋势")
history_df = pd.DataFrame([{
    "时间": h["timestamp"],
    "综合评分": h["scores"]["total"],
    "主机评分": h["scores"]["host"],
    "MySQL评分": h["scores"]["mysql"],
    "Redis评分": h["scores"]["redis"],
    "服务评分": h["scores"]["services"],
    "ES评分": h["scores"]["es"]
} for h in history])

fig = px.line(history_df, x="时间", y=["综合评分", "主机评分", "MySQL评分", "Redis评分", "服务评分", "ES评分"],
              title="过去24小时评分趋势")
st.plotly_chart(fig, use_container_width=True)

# 详细信息选项卡
tab1, tab2, tab3, tab4, tab5 = st.tabs(["主机指标", "MySQL指标", "Redis指标", "服务状态", "ES指标"])

with tab1:
    st.subheader("主机监控指标")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("CPU使用率", f"{data['host']['cpu_usage']:.1f}%")
        status_indicator(data['host']['cpu_usage'])

    with col2:
        st.metric("内存使用率", f"{data['host']['memory_usage']:.1f}%")
        status_indicator(data['host']['memory_usage'])

    with col3:
        st.metric("磁盘使用率", f"{data['host']['disk_usage']:.1f}%")
        status_indicator(data['host']['disk_usage'])

    # 主机指标趋势
    host_history = pd.DataFrame([{
        "时间": h["timestamp"],
        "CPU使用率": h["data"]["host"]["cpu_usage"],
        "内存使用率": h["data"]["host"]["memory_usage"],
        "磁盘使用率": h["data"]["host"]["disk_usage"]
    } for h in history])

    fig = px.line(host_history, x="时间", y=["CPU使用率", "内存使用率", "磁盘使用率"],
                  title="主机指标趋势")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("MySQL监控指标")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("连接数", data['mysql']['connections'])
        status_indicator(data['mysql']['connections'], [150, 180])

    with col2:
        st.metric("查询数", data['mysql']['queries'])

    with col3:
        st.metric("慢查询数", data['mysql']['slow_queries'])
        status_indicator(data['mysql']['slow_queries'], [5, 10])

    # MySQL指标趋势
    mysql_history = pd.DataFrame([{
        "时间": h["timestamp"],
        "连接数": h["data"]["mysql"]["connections"],
        "查询数": h["data"]["mysql"]["queries"],
        "慢查询数": h["data"]["mysql"]["slow_queries"]
    } for h in history])

    fig = px.line(mysql_history, x="时间", y=["连接数", "查询数", "慢查询数"],
                  title="MySQL指标趋势")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Redis监控指标")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("命中率", f"{data['redis']['hit_rate']:.1f}%")
        status_indicator(data['redis']['hit_rate'], [90, 95])

    with col2:
        st.metric("连接数", data['redis']['connections'])
        status_indicator(data['redis']['connections'], [70, 85])

    with col3:
        st.metric("内存使用(MB)", f"{data['redis']['memory_used']:.1f}")
        status_indicator(data['redis']['memory_used'], [1500, 1800])

    # Redis指标趋势
    redis_history = pd.DataFrame([{
        "时间": h["timestamp"],
        "命中率": h["data"]["redis"]["hit_rate"],
        "连接数": h["data"]["redis"]["connections"],
        "内存使用": h["data"]["redis"]["memory_used"]
    } for h in history])

    fig = px.line(redis_history, x="时间", y=["命中率", "连接数", "内存使用"],
                  title="Redis指标趋势")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("服务状态监控")

    service_status = data['services']
    status_map = {0: "正常", 1: "异常", 2: "警告"}
    status_color = {0: "green", 1: "red", 2: "orange"}

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"**API服务**: <span style='color:{status_color[service_status['api_service']]}'>"
                    f"{status_map[service_status['api_service']]}</span>",
                    unsafe_allow_html=True)

    with col2:
        st.markdown(f"**认证服务**: <span style='color:{status_color[service_status['auth_service']]}'>"
                    f"{status_map[service_status['auth_service']]}</span>",
                    unsafe_allow_html=True)

    with col3:
        st.markdown(f"**数据库服务**: <span style='color:{status_color[service_status['database_service']]}'>"
                    f"{status_map[service_status['database_service']]}</span>",
                    unsafe_allow_html=True)

    with col4:
        st.markdown(f"**缓存服务**: <span style='color:{status_color[service_status['cache_service']]}'>"
                    f"{status_map[service_status['cache_service']]}</span>",
                    unsafe_allow_html=True)

    # 服务状态历史
    service_history = pd.DataFrame([{
        "时间": h["timestamp"],
        "API服务": h["data"]["services"]["api_service"],
        "认证服务": h["data"]["services"]["auth_service"],
        "数据库服务": h["data"]["services"]["database_service"],
        "缓存服务": h["data"]["services"]["cache_service"]
    } for h in history])

    fig = px.line(service_history, x="时间", y=["API服务", "认证服务", "数据库服务", "缓存服务"],
                  title="服务状态历史（0=正常, 1=异常, 2=警告）")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Elasticsearch监控指标")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("CPU使用率", f"{data['es']['cpu_usage']:.1f}%")
        status_indicator(data['es']['cpu_usage'])

    with col2:
        st.metric("内存使用率", f"{data['es']['memory_usage']:.1f}%")
        status_indicator(data['es']['memory_usage'])

    with col3:
        st.metric("磁盘使用率", f"{data['es']['disk_usage']:.1f}%")
        status_indicator(data['es']['disk_usage'])

    # ES指标趋势
    es_history = pd.DataFrame([{
        "时间": h["timestamp"],
        "CPU使用率": h["data"]["es"]["cpu_usage"],
        "内存使用率": h["data"]["es"]["memory_usage"],
        "磁盘使用率": h["data"]["es"]["disk_usage"]
    } for h in history])

    fig = px.line(es_history, x="时间", y=["CPU使用率", "内存使用率", "磁盘使用率"],
                  title="ES指标趋势")
    st.plotly_chart(fig, use_container_width=True)

# 底部信息
st.markdown("---")
st.markdown(f"最后更新: {data['timestamp']} | 数据来源: 监控系统API")