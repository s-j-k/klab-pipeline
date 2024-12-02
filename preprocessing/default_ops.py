import suite2p


def default_ops():
    ops = suite2p.default_ops()
    # ---------- Main setting ----------
    ops['nplanes'] = 1
    ops['nchannels'] = 2
    ops['functional_chan'] = 2
    ops['tau'] = 0.7
    # 0.7 for GCaMP6f,
    # 0.7 for jrgeco https://github.com/MouseLand/suite2p/issues/233
    # 1.0 for GCaMP6m,
    # 1.25-1.5 for GCaMP6s,
    # ops['frames_include'] = 500

    # ---------- file io ----------
    ops['delete_bin'] = False
    ops['save_folder'] = 'suite2p'
    ops['move_bin'] = True

    # ---------- output setting ----------

    # ---------- registration ----------
    ops['align_by_chan'] = 2
    ops['batch_size'] = 1000

    # ---------- cellpose ----------
    ops['anatomical_only'] = 3
    ops['connected'] = True
    ops['diameter'] = 0

    return ops
