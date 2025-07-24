from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://127.0.0.1:8000/base")
response = remote_chain.invoke({
            "domain": "金融",
            "profession": "金融分析员",
            "do": "评估未来半年中国的金融市场情况，分析市场趋势，预测未来的市场变化",
            "material": "无",
            "question": "以表格的形式输出",
        })

if __name__ == '__main__':
    print(response)
