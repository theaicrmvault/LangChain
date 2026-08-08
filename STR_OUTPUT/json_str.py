# Import LangChain's OpenAI chat-model integration.
from langchain_openai import ChatOpenAI

# Utility for loading configuration values from environment variables.
from dotenv import load_dotenv

# Load environment variables such as the OpenAI API key
# from the .env file instead of hardcoding credentials.
load_dotenv()


# Define the expected structure of the model's response.
# This JSON Schema acts as a contract between the application
# and the LLM, specifying the fields, data types, descriptions,
# and allowed values.
json_schema = {
    # Required by LangChain when converting a JSON Schema
    # into a structured-output schema.
    "title": "ReviewInfo",
    "type": "object",
    "properties": {
        # Optional reviewer name.
        # The value can either be a string or null.
        "reviewer": {
            "type": ["string", "null"],
            "description": "The name of the person writing the review",
        },
        # Optional business/shop name.
        "name": {
            "type": ["string", "null"],
            "description": "The name of the shop or business being reviewed",
        },
        # Required short summary of the review.
        "summary": {"type": "string", "description": "A brief summary of the review"},
        # Required numerical rating given to the business.
        "rating": {
            "type": "integer",
            "description": "The rating given to the shop or business",
        },
        # List of positive aspects identified from the review.
        "pros": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The positive aspects of the shop or business",
        },
        # List of negative aspects identified from the review.
        "cons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The negative aspects of the shop or business",
        },
        # Restrict sentiment to one of three predefined values.
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "The overall sentiment of the review",
        },
    },
    # Fields that the model must return in the structured response.
    # reviewer and name are intentionally optional.
    "required": ["summary", "rating", "pros", "cons", "sentiment"],
}


# Initialize the chat model.
# The model will be used to extract structured information
# from an unstructured customer review.
llm = ChatOpenAI(model="gpt-5.6", temperature=1)


# Convert the regular chat model into a structured-output model.
# The schema defines the expected structure of the model's response,
# allowing the application to work with predictable fields instead
# of parsing free-form natural-language text.
model = llm.with_structured_output(json_schema)


# Unstructured input containing business information and
# customer feedback.
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
# The model extracts the relevant information and returns data
# conforming to the defined schema.
llm_response = model.invoke(review_text)


# Display the structured response.
print(llm_response)
