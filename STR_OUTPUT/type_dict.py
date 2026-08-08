# TypedDict defines the expected dictionary structure.
# Annotated attaches descriptions/metadata to individual fields.
# Optional allows a field to contain None.
# Literal restricts a field to predefined values.
from typing import TypedDict, Annotated, Optional, Literal

# LangChain integrations for Hugging Face hosted models.
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Utility for loading environment variables from the .env file.
from dotenv import load_dotenv

# Load environment variables such as Hugging Face API credentials.
load_dotenv()


# Define the expected structure of the LLM's response.
# TypedDict describes the keys and expected Python types,
# while Annotated provides descriptions that help define
# the meaning of each field.
class Review(TypedDict):

    # Optional reviewer name.
    reviewer: Annotated[Optional[str], "The name of the person writing the review"]

    # Optional business/shop name.
    name: Annotated[Optional[str], "The name of the shop or business being reviewed"]

    # Short summary of the review.
    summary: Annotated[str, "A brief summary of the review"]

    # Numerical rating given to the business.
    rating: Annotated[int, "The rating given to the shop or business"]

    # Positive aspects extracted from the review.
    pros: Annotated[list[str], "The positive aspects of the shop or business"]

    # Negative aspects extracted from the review.
    cons: Annotated[list[str], "The negative aspects of the shop or business"]

    # Restrict sentiment to predefined values.
    sentiment: Annotated[
        Literal["positive", "negative", "neutral"],
        "The overall sentiment of the review",
    ]


# Configure the Hugging Face hosted inference endpoint.
# repo_id identifies the model that will process the request.
llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct", task="text-generation")


# Wrap the Hugging Face endpoint with LangChain's
# standardized chat-model interface.
model = ChatHuggingFace(llm=llm)


# Configure the model to return data matching the Review schema
# instead of returning free-form natural-language text.
structured_review = model.with_structured_output(Review)


# Unstructured customer review that will be converted
# into the predefined Review structure.
review_text = """
Anil Atta Chakki Mall, located in Akurdi, Pune, is highly rated
with a 4.8-star average across thousands of customer reviews
on Google and Justdial. Customers praise the store for its wide
variety of domestic flour mills, durable build quality, free
installation, and helpful customer service.

Overview & Ratings
Platforms: Google and Justdial
Average Rating: 4.8 / 5 stars (based on over 3,000 combined reviews)
Location: Near Vegetable Market, Akurdi, Pune

Customer Feedback Pros
Product Range: Excellent variety of domestic and commercial
kitchen appliances including automatic flour mills, wet grinders,
and oil makers.

Service: Appreciated for knowledgeable staff, free home delivery,
and complimentary installation services in select areas.

Performance: Units like their 1 HP and 2 HP automatic domestic
models are noted for working efficiently with low power consumption
and manageable noise levels.

Customer Feedback Cons
After-Sales Support: A few isolated reviews mention delays during
peak festival seasons for servicing or repairs.

Pricing & Delivery: Delivery timelines outside immediate local
limits can sometimes vary depending on stock availability.
"""


# Invoke the structured-output model.
# The returned result is represented as a dictionary-like object
# containing the fields defined in the Review TypedDict.
result = structured_review.invoke(review_text)


# Access individual fields using dictionary-style key access.
print(result["reviewer"])
print(result["name"])
print(result["summary"])
print(result["rating"])
print(result["pros"])
print(result["cons"])
print(result["sentiment"])
