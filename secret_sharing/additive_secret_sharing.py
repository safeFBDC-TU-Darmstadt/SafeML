import util


def create_shares(secret, num_shares: int, random_fct):
    """
    Creates additive secret shares for an input ``secret``.

    Parameters
    ----------
    secret:
        Floating point tensor to create additive secret shares for.
    num_shares: int
        Number of secret shares to create.
    random_fct:
        Function to create random values or tensors in the representation of the input.
    fpr: bool
        Toggle `floating point` representation (False) or `fixed point` representation (True).

    Returns
    -------
    list
        A list of additive secret shares.
    """

    shares = []
    for i in range(num_shares-1):
        share = (util.constants.share_min - util.constants.share_max) * random_fct(secret.shape) + util.constants.share_max
        shares.append(share)
    shares.append(secret - sum(shares))

    return shares
