from tavily import TavilyClient

tavily_client = TavilyClient(api_key="tvly_API_KEY")
response = tavily_client.search("Who is Leo Messi?")

print(response)