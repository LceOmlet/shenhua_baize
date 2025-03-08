import streamlit as st

def main():
    st.title("多模态信息提取器")
    
    uploaded_files = st.file_uploader(
        "拖拽文件到这里 (图片/音频/文本)",
        type=["jpg", "png", "wav", "mp3", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            file_type = file.type.split('/')[0]
            st.write(f"已上传文件: {file.name} ({file_type})")

if __name__ == "__main__":
    main()
