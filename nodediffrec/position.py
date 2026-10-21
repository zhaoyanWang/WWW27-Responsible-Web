import torch, math
def create_positional_embeddings(seq_len, emb_dim):
    """Create positional embeddings."""
    # Initialize the matrix with zeros
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))

    # Calculate positional encodings
    positional_embeddings = torch.zeros(seq_len, emb_dim)
    positional_embeddings[:, 0::2] = torch.sin(position * div_term)
    positional_embeddings[:, 1::2] = torch.cos(position * div_term)

    return positional_embeddings
