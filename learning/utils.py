# get text clip encoding 

def get_text_embeddings():
    """
    Get text embeddings from clip language model
    """
    ALL_SEMANTIC_TYPES = [f"{v_desc}_{h_desc}_{cabinet_type}" for v_desc in ["", "bottom", "second-bottom", "middle", "second-top", "top"] for h_desc in ["", "right", "second-right", "middle", "second-left", "left"] for cabinet_type in ["drawer", "door"]]

    ALL_SEMANTIC_TYPES = [f"{v_desc}_{h_desc}_{cabinet_type}" for v_desc in ["", "bottom", "second-bottom", "middle", "second-top", "top"] for h_desc in ["", "right", "second-right", "middle", "second-left", "left"] for cabinet_type in ["drawer", "door"]]

    all_texts = [t.replace("_"," ").replace("-"," ").replace("  ", " ").strip() for t in ALL_SEMANTIC_TYPES]

    # all_texts

    from transformers import CLIPTokenizer, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    inputs = tokenizer(all_texts, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)

    text2feature = {all_texts[i]: text_features[i].data for i in range(72)}

    import pickle
    # save dictionary to pickle file
    with open('text2clip_feature.pickle', 'wb') as file:
        pickle.dump(text2feature, file, protocol=pickle.HIGHEST_PROTOCOL)
    

