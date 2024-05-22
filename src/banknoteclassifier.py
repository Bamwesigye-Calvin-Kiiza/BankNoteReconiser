import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import BatchNormalization
from groq import Groq
# Load your pre-trained model
model = load_model('./ResNet152V2.h5', custom_objects={'BatchNormalization': BatchNormalization})

# Function to preprocess image
def preprocess_image(image, target_size):
    """Preprocess the image by cropping, resizing, and normalizing."""
    # Convert the image to a PIL image for cropping and resizing
    image = Image.fromarray(np.uint8(image))

    # Crop the image to a square with the smallest dimension
    min_length = min(image.size)
    left = (image.width - min_length) / 2
    top = (image.height - min_length) / 2
    right = (image.width + min_length) / 2
    bottom = (image.height + min_length) / 2

    image = image.crop((left, top, right, bottom))

    # Resize the image to the target size
    image = image.resize((target_size, target_size), Image.BILINEAR)

    # Convert the image back to a numpy array
    image = img_to_array(image)

    # Normalize the image
    image = (image - 127.5) / 127.5

    return image

# Function to predict image with probability
def predict_image_with_probability(img):
    target_size = 224  # Adjust to your model's input size
    
    # Preprocess the image
    preprocessed_img = preprocess_image(img, target_size=target_size)
    
    # Add batch dimension
    x = np.expand_dims(preprocessed_img, axis=0)

    # Predict the class of the image
    prediction = model.predict(x, batch_size=1)

    # Assuming the model is trained for binary classification (Counterfeit vs Genuine)
    # with the output layer having two neurons (softmax activation)
    prediction_result = 'Counterfeit' if prediction[0][0] > 0.6 else 'Genuine'
    prediction_probability = prediction[0][0] if prediction_result == 'Counterfeit' else prediction[0][1]
    
    return prediction_result, prediction_probability

# Function to convert PIL image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Function to display detailed information about Counterfeit and Genuine notes
def display_information():
    st.subheader("Counterfeit vs. Genuine Notes")
    st.markdown("""
    The production and circulation of counterfeit currency is illegal and can have serious consequences.
    It's important to be aware of the differences between counterfeit and genuine notes to avoid falling victim to fraud.
    Below are some key differences to look out for:
    - **Watermark**: Genuine notes typically have a watermark that is visible when held up to the light. Counterfeit notes may lack this feature or have a poorly reproduced watermark.
    - **Security Thread**: Genuine notes often have a security thread embedded in the paper that can be seen when held up to the light. Counterfeit notes may have a fake security thread printed on the surface.
    - **Microprinting**: Genuine notes may contain microprinting that is difficult to replicate. Counterfeit notes may have blurry or inconsistent microprinting.
    - **Paper Quality**: Genuine notes are printed on high-quality paper with specific textures and features. Counterfeit notes may feel different or lack the same level of detail.
    
    If you suspect that a note may be counterfeit, do not accept it and report it to the authorities.
    """)

# Function to record the prediction result along with the probability
def record_prediction(result, probability):
    df = pd.DataFrame({'Result': [result], 'Probability': [probability]})
    df.to_csv('predictions.csv', mode='a', header=False, index=False)

# Function to load the prediction data
def load_predictions():
    try:
        return pd.read_csv('predictions.csv', names=['Result', 'Probability'])
    except FileNotFoundError:
        return pd.DataFrame(columns=['Result', 'Probability'])

# Function to plot the results
def plot_results():
    df = load_predictions()
    if df.empty:
        st.write("No predictions made yet.")
    else:
        counts = df['Result'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))

        # Customize the bar plot
        sns.barplot(x=counts.index, y=counts.values, palette={'Genuine': 'lightgreen', 'Counterfeit': 'lightcoral'}, ax=ax)
        
        # Adding title and labels
        ax.set_title('Prediction Results', fontsize=18, fontweight='bold', color='navy')
        ax.set_xlabel('Class', fontsize=14, fontweight='bold', color='navy')
        ax.set_ylabel('Count', fontsize=14, fontweight='bold', color='navy')
        
        # Adding the count labels on top of the bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12, color='black')
        
        # Remove spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        
        # Enhance grid and background
        ax.grid(True, which='major', linestyle='--', linewidth='0.2', color='gray')
        ax.set_facecolor('#f7f7f7')
        fig.patch.set_facecolor('#f7f7f7')
        
        st.pyplot(fig)

        st.write(f"Total Genuine Predictions: {counts.get('Genuine', 0)}")
        st.write(f"Total Counterfeit Predictions: {counts.get('Counterfeit', 0)}")

        # Display probability insights
        display_probability_insights(df)

# Function to display insights about prediction probabilities
def display_probability_insights(df):
    st.subheader("Confidence level statistics")

    counterfeit_probs = df[df['Result'] == 'Counterfeit']['Probability']
    genuine_probs = df[df['Result'] == 'Genuine']['Probability']

    # Summary statistics in table
    st.write("### Summary Statistics")
    stats_data = {
        "Statistic": ["Mean", "Median", "Standard Deviation"],
        "Counterfeit": [
            f"{counterfeit_probs.mean():.2f}",
            f"{counterfeit_probs.median():.2f}",
            f"{counterfeit_probs.std():.2f}"
        ],
        "Genuine": [
            f"{genuine_probs.mean():.2f}",
            f"{genuine_probs.median():.2f}",
            f"{genuine_probs.std():.2f}"
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    st.table(stats_df)

        
    # Plotting distribution of probabilities
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(counterfeit_probs, bins=20, kde=True, color='red', label='Counterfeit', ax=ax)
    sns.histplot(genuine_probs, bins=20, kde=True, color='green', label='Genuine', ax=ax)

    # Remove spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    
    ax.set_title('Distribution of Prediction Probabilities', fontsize=18, fontweight='bold', color='navy')
    ax.set_xlabel('Probability', fontsize=14, fontweight='bold', color='navy')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold', color='navy')
    ax.legend()

    st.pyplot(fig)
    
# Function to generate AI response
def chat_with_Assistant(messages):
    '''
    Generates replies about counterfeit notes 
    Args:
        messages (list of dict): List of message dicts with "role" and "content"
    Returns:
        generated message 
    '''
    client = Groq(
        api_key="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"
    )
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )
    script = chat_completion.choices[0].message.content
    script_without_asterisks = script.replace("**", "")
    return script_without_asterisks
# Streamlit app with sidebar
st.sidebar.title("Menu")

# Sidebar buttons with an overlay color of light blue
detector_button = st.sidebar.button("Detector", key="detector", on_click=lambda: st.session_state.update({"selected_option": "Detector"}))
security_features_button = st.sidebar.button("Security Features", key="security_features", on_click=lambda: st.session_state.update({"selected_option": "Security Features"}))
results_button = st.sidebar.button("Results", key="results", on_click=lambda: st.session_state.update({"selected_option": "Results"}))
results_button = st.sidebar.button("Chat with AI-Calvin", key="Chat with AI-Calvin", on_click=lambda: st.session_state.update({"selected_option": "Chat with AI-Calvin"}))
# Default to Detector
if "selected_option" not in st.session_state:
    st.session_state.selected_option = "Detector"


if st.session_state.selected_option == "Detector":
    st.title("Counterfeit Detector")
    st.write("Upload images to classify them as 'Counterfeit' or 'Genuine'.")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Add custom CSS for rounded corners
    st.markdown(
        """
        <style>
        .rounded-img {
            border-radius: 15px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if uploaded_files:
        st.write("Classified images")

        # Group images into rows of at most 3 columns
        for i in range(0, len(uploaded_files), 3):
            cols = st.columns(3)

            for j in range(3):
                if i + j < len(uploaded_files):
                    uploaded_file = uploaded_files[i + j]
                    image = Image.open(uploaded_file)

                    # Convert image to base64
                    img_str = image_to_base64(image)

                    with cols[j]:
                        # Display image with rounded corners
                        st.markdown(
                            f'<img src="data:image/png;base64,{img_str}" class="rounded-img" width="100%">',
                            unsafe_allow_html=True
                        )
                        prediction_result, prediction_probability = predict_image_with_probability(image)

                        if prediction_result == "Counterfeit":
                            st.error("**Counterfeit**")
                            record_prediction('Counterfeit', prediction_probability)
                        else:
                            st.success("**Genuine**")
                            record_prediction('Genuine', prediction_probability)

elif st.session_state.selected_option == "Security Features":
    display_information()

elif st.session_state.selected_option == "Results":
    st.title("Results")
    plot_results()

elif st.session_state.selected_option == "Chat with AI-Calvin":
    st.title("Talk to AI-Calvin")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Input field for user to enter message
    user_input = st.text_input("You:", key="user_input")

    # Button to send message
    if st.button("Send"):
        # Check if user input is provided
        if user_input:
            # Append user message to conversation
            st.session_state.conversation.append({"role": "user", "content": user_input})
            
            # Generate AI response
            script = chat_with_Assistant(st.session_state.conversation)
            
            # Append AI response to conversation
            st.session_state.conversation.append({"role": "assistant", "content": script})

    # Display conversation history in reverse order
    if st.session_state.conversation:
        for message in reversed(st.session_state.conversation):
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI-Calvin:** {message['content']}")
