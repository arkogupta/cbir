import faiss
from feature_extractor import get_feature_file
"""
    creates a faiss index object
    updated to create index without explicitely having dictionary in mem
    --should increase speed--
"""


def build_index(feature_file):
    xb = get_feature_file(feature_file)
    m,dim = xb.shape

    # idx = faiss.IndexFlatL2(dim)
    idx = faiss.IndexFlatIP(dim)
    idx.add(xb)
    # return idx

    # for performance tester
    return xb,idx


