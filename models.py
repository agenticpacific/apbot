from langchain_openai import ChatOpenAI

# get optimal, preferred model that is available for use.
# in order of preference.
model_list = [
    #"moonshotai/kimi-k2.5",  # 1T
    #"z-ai/glm5",  # 744b
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3.5-122b-a10b",
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-120b",
]


def get_optimal_nvidia_model(NVIDIA_API_KEY) -> str:
    for model_name in model_list:
        try:
            llm_model_nvidia = ChatOpenAI(
                model=model_name,
                api_key=NVIDIA_API_KEY,
                base_url="https://integrate.api.nvidia.com/v1",
                use_responses_api=False,
                stream_usage=False,
                max_retries=2,
                timeout=2,
            )
            # test if the model works by making a simple call
            msg = [
                ("user", "hello"),
            ]
            ai_msg = llm_model_nvidia.invoke(msg)
            resp = ai_msg.text.strip()
            # print(resp)
            if resp is not None:
                if len(resp) > 0:
                    print(f"Model {model_name} Is Available: {resp}")
                    return model_name
        except Exception as e:
            print(f"Model {model_name} Not Available. Error: {e}")
    return None


# test
# print(get_optimal_nvidia_model())
