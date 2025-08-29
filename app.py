import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from text_cleaner import clean_text, clean_text_spacy
from nlp_functions import show_wordcloud , plot_top_ngrams_bar_chart,detect_emotions,detect_overall_sentiment_avg,classify_custom,summarize_large_text

st.title("INTERACTIVE TEXT ANALYSIS PLATFORM")
st.divider()

a = st.sidebar.radio("SELECT ONE:", ["Process Textual data","Process Csv file"])

if a=="Process Textual data":
    st.header("Input your textual data.")
    text= st.text_area("Enter your text", height=150)

    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter your text")
        else:
            #clean and processing
            cleaned= clean_text(text)
            tokens= clean_text_spacy(cleaned)

            # WORD CLOUD
            if tokens:
                st.subheader("Word Cloud")
                wc_plot = show_wordcloud(tokens)
                st.pyplot(wc_plot)
            st.divider()

            # N-GRAM ANALYSIS.
            st.subheader("N-GRAM ANALYSIS")
            plot_top_ngrams_bar_chart(tokens, gram_n=3)
            st.divider()


            #EMOTION DETECTION
            st.subheader("EMOTION DETECTION.")
            top_emotions_df= detect_emotions(text)
            max_index = top_emotions_df["Score"].idxmax()
            Emotion = top_emotions_df.loc[max_index, "Emotion"]
            Score = top_emotions_df.loc[max_index, "Score"]
            st.write(f"Predicted Emotion :- {Emotion}, with {Score * 100}% confidence")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Top 5 Emotions:-")
                st.dataframe(top_emotions_df)
            with col2:
                st.markdown("Visualizing  through Bar Chart")
                fig = px.bar(top_emotions_df, x="Emotion", y="Score", color="Emotion"
                            )
                fig.update_layout(
                    template='plotly_white',
                    height=290,  # fix the height of the plot to match the table
                )
                st.plotly_chart(fig)


            # SENTIMENTAL ANALYSIS
            st.subheader("SENTIMENT DETECTION")
            result= detect_overall_sentiment_avg(text)
            if "error" in result:
                st.write("Error:", result["error"])
            else:
                st.write("Overall Sentiment:", result["overall_sentiment"])
                st.write("Average Scores:",
                         pd.DataFrame(list(result['average_scores'].items()), columns=['Emotion', 'Score']))
            st.divider()


            #TONE OF SPEECH DETECTION.
            st.subheader("TONE OF SPEECH DETECTION.")
            output= classify_custom(text)
            col1, col2=st.columns(2)
            with col1:
                st.markdown(f"Predicted  : {output['predicted_category']}, score : {output['score']}")
                st.write("Other Top Predicted Category.")
                for label, score in output["all_categories"][1:6]:
                    st.write(f"Label :- {label}, Score:- {score}")

            with col2:
                labels=[]
                scores=[]
                for label, score in output["all_categories"][1:6]:
                    labels.append(label)
                    scores.append(score)

                fig= px.bar(x=labels, y= scores, color=labels, title="Other Top  5 Predicted Category.",
                            height=400
                            )
                st.plotly_chart(fig)
            st.divider()


            # SUMMARY GENERATION
            st.subheader("SUMMARY GENERATION.")
            output= summarize_large_text(text)
            st.write(output)








if a=="Process Csv file":
    st.header("Upload your CSV file.")
    uploaded_file= st.file_uploader("Choose anm Csv file", type="csv")

    if uploaded_file is not None:
        df= pd.read_csv(uploaded_file)
        st.success("File Uploaded Successfully")
        st.divider()

        st.header("Choose filtering opetion.")

        # user selecting column to filter data
        column_name= st.selectbox("Select an column on which basis you want to filter the table", df.columns)

        #selecting unique values
        unique_vals= df[column_name].dropna().unique()
        selected_value= st.multiselect(f"Please choose value(s)  from {column_name}", unique_vals)

        #select the column that is textual column
        text_processing_column= st.selectbox("Select column for text analsis.", df.columns)

        # filtering
        if selected_value:
           filtered_df=  df[df[column_name].isin(selected_value)]
           filtered_df= filtered_df[text_processing_column]
           st.subheader("filtered Data.")
           st.dataframe(filtered_df)
           st.divider()
           text = " ".join(filtered_df.dropna().astype(str))


           # CLEANING OF TEXT
           cleaned = clean_text(text)
           tokens = clean_text_spacy(cleaned)
           st.subheader("Cleaned and Lemmitized Text.")
           st.write(" ".join(tokens) if tokens else "No meaning-full tokens Extracted")

           # WORD CLOUD
           if tokens:
               st.subheader("Word Cloud")
               wc_plot = show_wordcloud(tokens)
               st.pyplot(
                   wc_plot)  # st.pyplot is a function provided by Streamlit to display Matplotlib plots in a Streamlit app.
           st.divider()

           # N-GRAM ANALYSIS
           st.subheader("N-GRAM ANALYSIS")
           plot_top_ngrams_bar_chart(tokens, gram_n=3)
           st.divider()

           # EMOTION DETECTION
           st.subheader("EMOTION DETECTION")
           top_emotions_df = detect_emotions(text)
           max_index = top_emotions_df['Score'].idxmax()
           Emotion = top_emotions_df.loc[max_index, 'Emotion']
           score = top_emotions_df.loc[max_index, 'Score']
           st.write(f"Preidcted Emotion(Tone):- {Emotion}, with {score}% confidence")
           col1, col2 = st.columns(2)  # Create two columns for side-by-side display
           with col1:
               # Display table of top 5 emotions
               st.markdown("Top 5 Emotions")
               st.dataframe(top_emotions_df)
           with col2:
               # Display Plotly chart
               st.markdown("Confidence Bar Chart")
               fig = px.bar(top_emotions_df, x="Emotion", y="Score", color="Emotion"
                            , text_auto='.4f')
               fig.update_layout(
                   template='plotly_white',
                   height=290,  # fix the height of the plot to match the table
               )
               st.plotly_chart(fig)

           st.divider()

           # SENTIMENT ANALYSIS
           st.subheader("SENTIMENT  DETECTION")
           result = detect_overall_sentiment_avg(text)
           if "error" in result:
               st.write("Error:", result["error"])
           else:
               st.write("Overall Sentiment:", result["overall_sentiment"])
               st.write("Average Scores:",
                        pd.DataFrame(list(result['average_scores'].items()), columns=['Emotion', 'Score']))
           st.divider()

           # TONE OF SPEECH CLASSIFICATION
           st.subheader("TONE OF SPEECH CLASSIFICATION.")
           output = classify_custom(text)
           col1, col2 = st.columns(2)
           with col1:
               st.markdown(f"#### 🔍 Predicted: **{output['predicted_category']} (Score: {output['score']:.2f})**")
               st.write("📊 Top Categories:")
               for label, score in output["all_categories"][:5]:  # Show top 3
                   st.write(" ")
                   st.write(f"  - {label}: {score:.2f}")

           with col2:
               labels = []
               scores = []
               for label, score in output["all_categories"][:5]:
                   labels.append(label)
                   scores.append(score)

               fig = px.bar(
                   x=labels,
                   y=scores,
                   color=labels,
                   title="Top 5 sentence type classification.",
                   labels={"Value": "Value Count"},
                   height=400
               )

               st.plotly_chart(fig)






