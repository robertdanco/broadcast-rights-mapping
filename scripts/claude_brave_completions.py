import anthropic
import requests
import json
from time import sleep
from typing import List, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel


# Config class for structured output
class StructuredOutput(BaseModel):
    title: str
    summary: str
    key_points: List[str]
    confidence_score: float


class BraveSearchClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {"Accept": "application/json", "X-Subscription-Token": api_key}

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Perform a web search using Brave Search API"""
        try:
            response = requests.get(
                self.base_url,
                params={"q": query, "count": max_results},
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            sleep(1)  # Rate limiting
            return response.json().get("web", {}).get("results", [])
        except Exception as e:
            print(f"Search error: {e}")
            return []


class SearchAugmentedGenerator:
    def __init__(self, anthropic_api_key: str, brave_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.search_client = BraveSearchClient(brave_api_key)

    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a string for the prompt"""
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f'<result index="{i}">\n'
                f'<url>{result.get("url", "")}</url>\n'
                f'<title>{result.get("title", "")}</title>\n'
                f'<description>{result.get("description", "")}</description>\n'
                f"</result>"
            )
        return "\n".join(formatted_results)

    def generate_structured_response(self, query: str) -> StructuredOutput:
        """Generate a structured response using search results and Claude"""
        # Perform search
        search_results = self.search_client.search(query)
        formatted_results = self.format_search_results(search_results)

        # Create prompt
        prompt = f"""Please analyze these search results and provide a structured response:

{formatted_results}

Based on these search results, provide a response about "{query}" with:
1. A clear title
2. A brief summary
3. 3-5 key points
4. A confidence score (0-1) based on the search result quality and relevance

Return your response in this exact JSON format:
{{
    "title": "string",
    "summary": "string",
    "key_points": ["string"],
    "confidence_score": float
}}
"""

        # Get completion from Claude
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            # Parse response into structured output
            response_content = response.content[0].text
            response_json = json.loads(response_content)
            return StructuredOutput(**response_json)

        except Exception as e:
            print(f"Generation error: {e}")
            # Return a default structured output in case of error
            return StructuredOutput(
                title="Error in Generation",
                summary="Failed to generate response",
                key_points=["Error occurred during generation"],
                confidence_score=0.0,
            )


def main():
    # Replace with your API keys
    ANTHROPIC_API_KEY = "your_anthropic_api_key"
    BRAVE_API_KEY = "your_brave_api_key"

    generator = SearchAugmentedGenerator(ANTHROPIC_API_KEY, BRAVE_API_KEY)

    # Example usage
    query = "What are the latest developments in quantum computing?"
    result = generator.generate_structured_response(query)

    # Print formatted output
    print("\nStructured Response:")
    print(f"Title: {result.title}")
    print(f"Summary: {result.summary}")
    print("\nKey Points:")
    for point in result.key_points:
        print(f"- {point}")
    print(f"\nConfidence Score: {result.confidence_score}")


if __name__ == "__main__":
    main()
