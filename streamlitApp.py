import streamlit as st
from sample import caption_generation


def main():
    st.write("# Image Caption Generator")
    st.write("Drag the image and the caption will be generated")
    with st.form(key="caption"):
        st.write("## Caption generator")
        img = st.file_uploader("Drag the Image", type=['jpg', 'png', 'jpeg'])
        # st.write(type(img), img.name)
        submit = st.form_submit_button(label='Submit')
    if submit:
        caption_dict = caption_generation(img)
        st.image(img)
        # st.write("Image name: ", img_name)
        # st.write("Caption: ", caption)
        st.write(caption_dict)
    else:
        st.stop()


if __name__ == '__main__':
    main()

