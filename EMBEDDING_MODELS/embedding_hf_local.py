from langchain_huggingface import HuggingFaceEmbeddings

text = ["The theory of relativity, developed by Albert Einstein ",
        "revolutionized our understanding of space and time. It consists of two main parts: special relativity and general relativity. Special relativity, published in 1905, introduced the idea that the laws of physics are the same for all non-accelerating observers",
        "and that the speed of light is constant regardless of the motion of the light source. This led to the famous equation E=mc^2, which shows the relationship ",
        "between energy and mass. General relativity, published in 1915, expanded on this by describing how gravity is not just a force between masses but a curvature of spacetime caused by mass and energy. This theory has been confirmed by many experiments and observations, such as the bending of light."]
        
        

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

vector  = embeddings.embed_documents(text)
print(str(vector))