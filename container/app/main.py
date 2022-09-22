from enum import Enum
from io import BytesIO, StringIO
from typing import Union

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import nlplot as nlplot

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

FILE_TYPES = ["csv", "py", "png", "jpg"]
st.set_option('deprecation.showfileUploaderEncoding', False)


class FileType(Enum):
    """Used to distinguish between file types"""

    IMAGE = "Image"
    CSV = "csv"
    PYTHON = "Python"


@st.cache
def load_data(file):
    """データをロードする関数"""
    data = pd.read_csv(file)
    return data


def get_file_type(file: Union[BytesIO, StringIO]) -> FileType:
    """The file uploader widget does not provide information on the type of file uploaded so we have
    to guess using rules or ML

    I've implemented rules for now :-)

    Arguments:
        file {Union[BytesIO, StringIO]} -- The file uploaded

    Returns:
        FileType -- A best guess of the file type
    """

    if isinstance(file, BytesIO):
        return FileType.IMAGE
    content = file.getvalue()
    if (
        content.startswith('"""')
        or "import" in content
        or "from " in content
        or "def " in content
        or "class " in content
        or "print(" in content
    ):
        return FileType.PYTHON

    return FileType.CSV


def main():
    """Run this function to display the Streamlit app"""
    # st.info(__doc__)
    # st.markdown(STYLE, unsafe_allow_html=True)

    st.header('nlplot デモアプリ')

    ##############
    # サイドバー
    ##############
    st.sidebar.markdown('## 描画するデータを選択')

    file = st.sidebar.file_uploader("Upload file", type=FILE_TYPES)
    if not file:
        st.sidebar.info("Please upload a file of type: " + ", ".join(FILE_TYPES))
        return

    # データのロード
    data = load_data(file)

    # 分析対象のカラムをセレクトボックスで選択
    selected_col = st.sidebar.selectbox(
        'カラム名：',
        data.columns.tolist()
    )
    # 表示ワードをセレクトボックスで選択
    selected_word = st.sidebar.selectbox(
        '表示するワード：',
        ['all'] + data['searched_for'].unique().tolist()
    )
    # N-gramで表示するN数を選択
    n_gram_num = st.sidebar.selectbox(
        'N-gram：',
        [1, 2, 3]
    )
    # 描画する図をプロット
    plot_type = st.sidebar.selectbox(
        '描画する図：',
        ['-', 'bar chart', 'tree map', 'histogram', 'wordcloud', 'co-occurrence network', 'sunburst chart']
    )

    # データフレームを表示するチェックボックス
    is_show_dataframe = st.sidebar.checkbox('Show dataframe')

    ##############
    # body
    ##############
    # ボタンを押したタイミングでbodyを描画
    if st.sidebar.button('Show'):
        # ボタン押されたときの処理
        st.write(f'### 選択したワード：{selected_word}')

        # データフレームの表示
        if is_show_dataframe:
            if selected_word == 'all':
                st.dataframe(data)
            else:
                st.dataframe(data.query('searched_for == @selected_word'))

        if plot_type != '-':
            # インスタンスの生成
            if selected_word == 'all':
                _df = data.copy()
            else:
                _df = data.query('searched_for == @selected_word').copy()
            npt = nlplot.NLPlot(_df, target_col=selected_col)

            # プロット
            if plot_type == 'bar chart':
                with st.spinner('Wait for it...'):
                    fig = npt.bar_ngram(
                        title='',
                        xaxis_label='word_count',
                        yaxis_label='word',
                        ngram=n_gram_num,
                        top_n=50,
                        stopwords=[],
                    )
                st.write(fig)
            elif plot_type == 'histogram':
                with st.spinner('Wait for it...'):
                    fig = npt.word_distribution(
                        title='',
                        xaxis_label='count',
                        yaxis_label='',
                        width=1000,
                        height=500,
                        color=None,
                        template='plotly',
                        bins=None,
                        save=False,
                    )
                st.write(fig)
            elif plot_type == 'tree map':
                with st.spinner('Wait for it...'):
                    fig = npt.treemap(
                        title='',
                        ngram=n_gram_num,
                        top_n=30,
                        stopwords=[],
                    )
                st.write(fig)
            elif plot_type == 'wordcloud':
                with st.spinner('Wait for it...'):
                    fig = npt.wordcloud(
                        width=1000,
                        height=600,
                        max_words=100,
                        max_font_size=100,
                        colormap='tab20_r',
                        mask_file=None,
                        save=False
                    )
                    plt.figure(figsize=(15, 25))
                    plt.imshow(fig, interpolation="bilinear")
                    plt.tight_layout()
                    plt.axis("off")
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
            elif plot_type == 'co-occurrence network':
                with st.spinner('Wait for it...'):
                    npt.build_graph(stopwords=[], min_edge_frequency=20)
                    fig = npt.co_network(
                        title='Co-occurrence network',
                        save=False
                    )
                    st.write(fig)
            elif plot_type == 'sunburst chart':
                with st.spinner('Wait for it...'):
                    npt.build_graph(stopwords=[], min_edge_frequency=40)
                    fig = npt.sunburst(
                        title='Sunburst chart',
                        colorscale=True,
                        save=False
                    )
                st.write(fig)

        else:
            st.write('サイドメニューで表示する条件を選択後[Show]ボタンを押下してください')
            pass


main()
