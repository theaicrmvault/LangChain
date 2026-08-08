# Pydantic provides the BaseModel and Field classes used to
# define and validate the structure of the LLM's response.
from pydantic import BaseModel, Field

# LangChain integration for OpenAI chat models.
from langchain_openai import ChatOpenAI

# Optional allows a field to contain either a value or None.
# Literal restricts a field to a predefined set of values.
from typing import Optional, Literal

# Utility for loading environment variables from the .env file.
from dotenv import load_dotenv

# Load environment variables such as the OpenAI API key.
load_dotenv()


# Define the expected structure of the review using Pydantic.
# This model acts as a typed contract for the LLM output.
class ReviewModel(BaseModel):

    # Optional reviewer name.
    # The LLM can return a string or None when the reviewer
    # information is not available.
    reviewer: Optional[str] = Field(
        description="The name of the person writing the review"
    )

    # Business or shop name.
    name: str = Field(description="The name of the shop or business being reviewed")

    # Short summary of the review.
    summary: str = Field(description="A brief summary of the review")

    # Numerical rating assigned to the business.
    rating: float = Field(description="The rating given to the shop or business")

    # Positive aspects extracted from the review.
    pros: list[str] = Field(description="The positive aspects of the shop or business")

    # Negative aspects extracted from the review.
    cons: list[str] = Field(description="The negative aspects of the shop or business")

    # Restrict sentiment to only the allowed values.
    # This prevents arbitrary sentiment values from being returned.
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The overall sentiment of the review",
        example="positive, negative, neutral",
    )


# Initialize the OpenAI chat model.
# temperature=0 is used when deterministic and consistent
# extraction is preferred over creative responses.
model = ChatOpenAI(model_name="gpt-5.6", temperature=0)


# Convert the regular chat model into a structured-output model.
# LangChain uses the Pydantic model as the output schema and
# returns the response as a validated ReviewModel object.
structured_review = model.with_structured_output(ReviewModel)


# Unstructured customer review that will be analyzed
# and converted into the ReviewModel structure.
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


# Send the unstructured review to the structured-output model.
# The result is expected to conform to the ReviewModel schema.
result = structured_review.invoke(review_text)


# The response is a Pydantic ReviewModel object.
# Unlike free-form LLM output, individual fields can be accessed
# directly using dot notation.
print(result)

print(result.reviewer)
print(result.name)
print(result.summary)
print(result.rating)
print(result.pros)
print(result.cons)
print(result.sentiment)
