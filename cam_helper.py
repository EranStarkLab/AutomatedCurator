__all__ = ['CMAP', 'hooked_backward', 'hook_acts', 'cam_acts', 'acts_scaled', 'grad_cam_acts', 'CAM_batch_compute',
           'batchify', 'itemize', 'get_list_items', 'get_batch', 'show_cam', 'cam_batch_plot_one_fig',
           'cam_batch_plot_multi_fig', 'i2o', 'lbl_dict']

# Cell
from fastai.imports import *
from fastai.basics import *
from fastai.callback.hook import *
from fastai.vision.data import get_grid
from tsai.data.core import TSTensor

# Cell
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm

# Cell
class CMAP():
    'There are 164 different palettes.'
    Accent = 'Accent'
    Accent_r = 'Accent_r'
    Blues = 'Blues'
    Blues_r = 'Blues_r'
    BrBG = 'BrBG'
    BrBG_r = 'BrBG_r'
    BuGn = 'BuGn'
    BuGn_r = 'BuGn_r'
    BuPu = 'BuPu'
    BuPu_r = 'BuPu_r'
    CMRmap = 'CMRmap'
    CMRmap_r = 'CMRmap_r'
    Dark2 = 'Dark2'
    Dark2_r = 'Dark2_r'
    GnBu = 'GnBu'
    GnBu_r = 'GnBu_r'
    Greens = 'Greens'
    Greens_r = 'Greens_r'
    Greys = 'Greys'
    Greys_r = 'Greys_r'
    OrRd = 'OrRd'
    OrRd_r = 'OrRd_r'
    Oranges = 'Oranges'
    Oranges_r = 'Oranges_r'
    PRGn = 'PRGn'
    PRGn_r = 'PRGn_r'
    Paired = 'Paired'
    Paired_r = 'Paired_r'
    Pastel1 = 'Pastel1'
    Pastel1_r = 'Pastel1_r'
    Pastel2 = 'Pastel2'
    Pastel2_r = 'Pastel2_r'
    PiYG = 'PiYG'
    PiYG_r = 'PiYG_r'
    PuBu = 'PuBu'
    PuBuGn = 'PuBuGn'
    PuBuGn_r = 'PuBuGn_r'
    PuBu_r = 'PuBu_r'
    PuOr = 'PuOr'
    PuOr_r = 'PuOr_r'
    PuRd = 'PuRd'
    PuRd_r = 'PuRd_r'
    Purples = 'Purples'
    Purples_r = 'Purples_r'
    RdBu = 'RdBu'
    RdBu_r = 'RdBu_r'
    RdGy = 'RdGy'
    RdGy_r = 'RdGy_r'
    RdPu = 'RdPu'
    RdPu_r = 'RdPu_r'
    RdYlBu = 'RdYlBu'
    RdYlBu_r = 'RdYlBu_r'
    RdYlGn = 'RdYlGn'
    RdYlGn_r = 'RdYlGn_r'
    Reds = 'Reds'
    Reds_r = 'Reds_r'
    Set1 = 'Set1'
    Set1_r = 'Set1_r'
    Set2 = 'Set2'
    Set2_r = 'Set2_r'
    Set3 = 'Set3'
    Set3_r = 'Set3_r'
    Spectral = 'Spectral'
    Spectral_r = 'Spectral_r'
    Wistia = 'Wistia'
    Wistia_r = 'Wistia_r'
    YlGn = 'YlGn'
    YlGnBu = 'YlGnBu'
    YlGnBu_r = 'YlGnBu_r'
    YlGn_r = 'YlGn_r'
    YlOrBr = 'YlOrBr'
    YlOrBr_r = 'YlOrBr_r'
    YlOrRd = 'YlOrRd'
    YlOrRd_r = 'YlOrRd_r'
    afmhot = 'afmhot'
    afmhot_r = 'afmhot_r'
    autumn = 'autumn'
    autumn_r = 'autumn_r'
    binary = 'binary'
    binary_r = 'binary_r'
    bone = 'bone'
    bone_r = 'bone_r'
    brg = 'brg'
    brg_r = 'brg_r'
    bwr = 'bwr'
    bwr_r = 'bwr_r'
    cividis = 'cividis'
    cividis_r = 'cividis_r'
    cool = 'cool'
    cool_r = 'cool_r'
    coolwarm = 'coolwarm'
    coolwarm_r = 'coolwarm_r'
    copper = 'copper'
    copper_r = 'copper_r'
    cubehelix = 'cubehelix'
    cubehelix_r = 'cubehelix_r'
    flag = 'flag'
    flag_r = 'flag_r'
    gist_earth = 'gist_earth'
    gist_earth_r = 'gist_earth_r'
    gist_gray = 'gist_gray'
    gist_gray_r = 'gist_gray_r'
    gist_heat = 'gist_heat'
    gist_heat_r = 'gist_heat_r'
    gist_ncar = 'gist_ncar'
    gist_ncar_r = 'gist_ncar_r'
    gist_rainbow = 'gist_rainbow'
    gist_rainbow_r = 'gist_rainbow_r'
    gist_stern = 'gist_stern'
    gist_stern_r = 'gist_stern_r'
    gist_yarg = 'gist_yarg'
    gist_yarg_r = 'gist_yarg_r'
    gnuplot = 'gnuplot'
    gnuplot2 = 'gnuplot2'
    gnuplot2_r = 'gnuplot2_r'
    gnuplot_r = 'gnuplot_r'
    gray = 'gray'
    gray_r = 'gray_r'
    hot = 'hot'
    hot_r = 'hot_r'
    hsv = 'hsv'
    hsv_r = 'hsv_r'
    inferno = 'inferno'
    inferno_r = 'inferno_r'
    jet = 'jet'
    jet_r = 'jet_r'
    magma = 'magma'
    magma_r = 'magma_r'
    nipy_spectral = 'nipy_spectral'
    nipy_spectral_r = 'nipy_spectral_r'
    ocean = 'ocean'
    ocean_r = 'ocean_r'
    pink = 'pink'
    pink_r = 'pink_r'
    plasma = 'plasma'
    plasma_r = 'plasma_r'
    prism = 'prism'
    prism_r = 'prism_r'
    rainbow = 'rainbow'
    rainbow_r = 'rainbow_r'
    seismic = 'seismic'
    seismic_r = 'seismic_r'
    spring = 'spring'
    spring_r = 'spring_r'
    summer = 'summer'
    summer_r = 'summer_r'
    tab10 = 'tab10'
    tab10_r = 'tab10_r'
    tab20 = 'tab20'
    tab20_r = 'tab20_r'
    tab20b = 'tab20b'
    tab20b_r = 'tab20b_r'
    tab20c = 'tab20c'
    tab20c_r = 'tab20c_r'
    terrain = 'terrain'
    terrain_r = 'terrain_r'
    twilight = 'twilight'
    twilight_r = 'twilight_r'
    twilight_shifted = 'twilight_shifted'
    twilight_shifted_r = 'twilight_shifted_r'
    viridis = 'viridis'
    viridis_r = 'viridis_r'
    winter = 'winter'
    winter_r = 'winter_r'

# Cell
def hooked_backward(x, y, model, layer):
    "A function hook to get access to both activation and gradient values of a given `model` at the layer number `layer`"
    xb= x[None, :] # xb = x.unsqueeze()
    with hook_output(model[layer]) as hook_a:
        with hook_output(model[layer], grad=True) as hook_g:
            preds = model(xb)
            preds[0,int(y)].backward()
    return hook_a,hook_g

# Cell
def hook_acts(x, y, model, layer):
    "A hook function to get access to activation values of a given `model` at the layer number `layer`"
    hook_a,hook_g = hooked_backward(x, y, model, layer)
    acts  = hook_a.stored[0].cpu()
    return acts

# Cell
def cam_acts(tseries, y, model, layer, reduction='mean', force_scale=True, scale_range=(0, 1)): # x.shape = [1, 150]
    "Compute raw CAM values. `reduction` : string. One of ['mean', 'median', 'max', 'mean_max']. 'mean_max' corresponds to       (mean + max)/2"

    # acts.shape = [128, 150]
    acts = hook_acts(tseries, y, model, layer)
    acts = acts.cpu()

    if reduction=='mean': acts = acts.mean(0)                                  # mean.shape = [150]
    if reduction=='median': acts = acts.median(0).values                       # mendia.shape = [150]
    if reduction=='max' : acts = acts.max(0).values                             # max.shape = [150]
    if reduction=='mean_max': acts = (acts.mean(0) + acts.max(0).values)/2 # max_mean.shape = [150]
    # print(f'Reduction: {reduction}')

    if force_scale==True:
        acts = (acts - acts.min())/(acts.max() - acts.min())*(scale_range[1] - scale_range[0]) + scale_range[0]
        return acts[None, :] # acts.shape = [1, 150]
    else:
        return acts[None, :] # acts.shape = [1, 150]

# store function name
cam_acts.name = r'CAM'

# Cell
def acts_scaled(acts, scale_range=(0, 1)):
    "Scale values between [scale_range[0]...scale_range[1]]. By default, it scales `acts` between 0 and 1"
    return (acts - acts.min())/(acts.max() - acts.min())*(scale_range[1] - scale_range[0]) + scale_range[0]


# Cell
def grad_cam_acts(tseries, y, model, layer, reduction='mean', force_scale=True, scale_range=(0, 1)): # x.shape = [1, 150]
    "Compute raw CAM values. `reduction` : string. One of ['mean', 'median', 'max', 'mean_max']. 'mean_max' corresponds       to (mean + max)/2"
    hook_a,hook_g = hooked_backward(tseries, y, model, layer)

    acts  = hook_a.stored[0].cpu()      # acts.shape = [128, 150]
    grad = hook_g.stored[0][0].cpu()    # grad.shape = [128, 150]

    # grad_chan.shape = [128]
    if reduction=='mean': grad_chan = grad.mean(1)
    if reduction=='median': grad_chan = grad.median(1).values
    if reduction=='max': grad_chan = grad.max(1).values
    if reduction=='mean_max': grad_chan = (grad.mean(1) + grad.max(1).values)/2
    # print(f'Reduction: {reduction}')

    mult = (acts*grad_chan[..., None])  # shape grad_chan[..., None] => [128, 1] => broadcast to => [128, 150]
    acts = mult.mean(0)                 # mean.shape = [150]

    if force_scale==True:
        acts = (acts - acts.min())/(acts.max() - acts.min())*(scale_range[1] - scale_range[0]) + scale_range[0]
        return acts[None, :]    # acts.shape = [1, 150]
    else:
        return acts[None, :]    # acts.shape = [1, 150]

# store function name
grad_cam_acts.name = r'GRAD-CAM'

# Cell
# # User defined CAM method
# def user_defined_cam_acts(tseries, y, model, layer, reduction='mean', force_scale=True, scale_range=(0, 1)):
# "Compute User-defined CAM values. `reduction` : string. One of ['mean', 'median', 'max', 'mean_max']. 'mean_max' corresponds to       (mean + max)/2"

#     # acts.shape = [128, 150]
#     acts = your_cam_method(tseries, y, model, layer)

#     if reduction=='mean': acts = acts.mean(0)                                  # mean.shape = [150]
#     if reduction=='median': acts = acts.median(0).values                       # mendia.shape = [150]
#     if reduction=='max' : acts = acts.max(0).values                             # max.shape = [150]
#     if reduction=='mean_max': acts = (acts.mean(0) + acts.max(0).values)/2 # max_mean.shape = [150]
#     # print(f'Reduction: {reduction}')

#     if force_scale==True:
#         acts = (acts - acts.min())/(acts.max() - acts.min())*(scale_range[1] - scale_range[0]) + scale_range[0]
#         return acts[None, :] # acts.shape = [1, 150]
#     else:
#         return acts[None, :] # acts.shape = [1, 150]

# # store function name
# cam_acts.name = r'user_defined_cam_acts'


# show_cam(batch, model, layer=5, i2o=i2o, func_cam=user_defined_cam_acts)

# Cell
@delegates(LineCollection.__init__)
def CAM_batch_compute(b, model, layer=5, func_cam=grad_cam_acts, reduction='mean',transform=None, force_scale=True, scale_range=(0, 1),**kwargs):

    'Compute either CAM for a list (b) of time series `tseries` .'

    ts_min_max = [1e10, -1e10]
    acts_min_max = [1e10, -1e10]
    tseries_list = []
    y_list = []
    acts_list = []
    idx = 1
    # Compute activation for each time series `tseries` here below
    # Process one time series at the time
    for item in b:
        # print(f'idx = {idx}')
        tseries, y = item
        y_list.append(y)
        acts = func_cam(tseries, y, model, layer=layer, reduction=reduction, force_scale=force_scale, scale_range=scale_range)

        # remove the first dimension : [1, 150] -> [150]
        acts = acts.squeeze().numpy()
        if transform is not None:
            tseries = transform(TSTensor(tseries.cpu()))
        
        tseries = tseries.squeeze().numpy()

        tseries_list.append(tseries)
        acts_list.append(acts)

        # set `tseries min and max
        min, max = tseries.min(), tseries.max()
        # print('min - max', min, max)
        if min<ts_min_max[0]: ts_min_max[0] = min
        if max>ts_min_max[1]: ts_min_max[1] = max
        # print('tsmin - tsmax', ts_min_max[0], ts_min_max[1])

        # set `tseries min and max
        min, max = acts.min(), acts.max()
        # print('min - max', min, max)
        if min<acts_min_max[0]: acts_min_max[0] = min
        if max>acts_min_max[1]: acts_min_max[1] = max
        # print('actsmin - actsmax', acts_min_max[0], acts_min_max[1])

        idx += 1

    return (tseries_list, acts_list, y_list, ts_min_max, acts_min_max)


# Cell
def batchify(dataset, idxs):
    'Return a list of items for the supplied dataset and idxs'
    tss = [dataset[i][0] for i in idxs]
    ys  = [dataset[i][1] for i in idxs]
    return (tss, ys)

# Cell
def itemize(batch):
    #take a batch and create a list of items. Each item represent a tuple of (tseries, y)
    tss, ys = batch
    b = [(ts, y) for ts,y in zip(tss, ys)]
    return b

# Cell
def get_list_items(dataset, idxs):
    'Return a list of items for the supplied dataset and idxs'
    list = [dataset[i] for i in idxs]
    return list

# Cell
def get_batch(dataset, idxs, func = noop):
    'Return a batch based on list of items from dataset at idxs'
    list_items = [func(dataset[i]) for i in idxs]
    tdl = TfmdDL(list_items, bs=len(idxs), num_workers=0)
    tdl.to(default_device())
    return tdl.one_batch()

# Cell
@delegates(LineCollection.__init__)
def show_cam(batch, model, layer=5, func_cam=cam_acts, reduction='mean', force_scale=True,
                    scale_range=(0, 1), cmap="Spectral_r", linewidth=4, linestyles='solid', alpha=1.0, scatter=False,
                    i2o=noop, figsize=None, multi_fig=False, confidence=None, transform=None, savefig=None, **kwargs):

    'Compute CAM using `func_cam` function, and plot a batch of colored time series `tseries`. The colors correspond to the         scaled CAM values. The time series are plot either on a single figure or on a multiple figures'

    # args = []
    if multi_fig==False:
        if figsize==None: figsize=(6,4)
        return cam_batch_plot_one_fig(batch, model, layer=layer, func_cam=func_cam, reduction=reduction, force_scale=force_scale,
                            scale_range=scale_range, cmap=cmap, linewidth=linewidth, linestyles=linestyles, alpha=alpha,
                            scatter=scatter, i2o=i2o, figsize=figsize, confidence=confidence, savefig=savefig,**kwargs)
    else:
        if figsize==None: figsize=(13,4)
    return cam_batch_plot_multi_fig(batch, model, layer=layer, func_cam=func_cam, reduction=reduction, force_scale=force_scale,                                             scale_range=scale_range, cmap=cmap, linewidth=linewidth, linestyles=linestyles, alpha=alpha, scatter=scatter, i2o=i2o,                           figsize=figsize,confidence=confidence,transform=transform, savefig=savefig, **kwargs)

# Cell
@delegates(LineCollection.__init__)
def cam_batch_plot_one_fig(batch, model, layer=5, func_cam=cam_acts, reduction='mean', force_scale=True,
                            scale_range=(0, 1), cmap="Spectral_r", linewidth=4, linestyles='solid', alpha=1.0, scatter=False,
                            i2o=noop, figsize=(6,4), confidence=None, savefig=None, **kwargs):

    'Compute CAM using `func_cam` function, and plot a batch of colored time series `tseries`. The colors correspond to the   scaled CAM values. The time series are plot on a single figure'

    """
    linestyles : string, tuple, optional
            Either one of [ 'solid' | 'dashed' | 'dashdot' | 'dotted' ], or
            a dash tuple. The dash tuple is:(offset, onoffseq)
            where ``onoffseq`` is an even length tuple of on and off ink in points.
    """

    fig = plt.figure(figsize=figsize)

    # batch use-cases
    # batch can be either:
    #   - a real batch meaning a tuple of (a list of tseries, a list of y)
    #   - a list of tuples (tseries, y) that we build from a dataset (a dataset item is a tuple of (tseries, y)
    #   - a single dataset item meaning a a tuple of (tseries, y)

    if not isinstance(batch, list):
        if len(batch[0].shape)==3:   # a real batch meaning a tuple of (a list of tseries, a list of y)
            b = itemize(batch)
        elif len(batch[0].shape)==2: # a single dataset item meaning a a tuple of (tseries, y)m
            b = [batch]
    else: b = batch # a list of tuples (tseries, y) that we build from a dataset (a dataset item is a tuple of (tseries, y)

    # b = _listify(b)
    tseries_list, acts_list, y_list, ts_min_max, acts_min_max = CAM_batch_compute(b, model, layer=layer, func_cam=func_cam,                                              reduction=reduction, force_scale=force_scale, scale_range=scale_range, **kwargs)

    levels = 254
    cmap = plt.get_cmap(cmap, lut=levels) #seismic
    if force_scale==True:
        colors = cmap(np.linspace(scale_range[0], scale_range[1], levels))
        norm = BoundaryNorm(np.linspace(scale_range[0], scale_range[1], levels+1), len(colors))
    else:
        colors = cmap(np.linspace(acts_min_max[0], acts_min_max[1], levels))
        norm = BoundaryNorm(np.linspace(acts_min_max[0], acts_min_max[1], levels+1), len(colors))

    # Plot activation `acts` (superimposed on the original time series `tseries)
    idx = 1
    for tseries, acts, y in zip(tseries_list, acts_list, y_list):
        t = np.arange(tseries.shape[-1])
        if scatter==True:
            # plt.figure(figsize=(13,4))
            # plt.subplot(1, 2, idx)
            plt.scatter(t, tseries, cmap=cmap, c = acts, linewidths=linewidth )
            idx += 1
        else:
            points = np.array([t, tseries]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            if idx==2: linestyles='dashed'
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, linestyles=linestyles, alpha=alpha,                                    **kwargs )
            lc.set_array(acts)
            lc.set_linewidth(linewidth)
            plt.gca().add_collection(lc)
            idx += 1

    plt.xlim(t.min(), t.max())
    plt.ylim(ts_min_max[0]*1.2, ts_min_max[1]*1.2)

    titles = [i2o(y) for y in y_list]
    title = ' - '.join(titles)
    if not hasattr(func_cam, 'name'): func_cam.name = str(func_cam)
    title =  f'[{title}] - {func_cam.name} - {reduction}'
    if confidence!=None: title = f'{title}\n confidence[0]'
    # print(f'Title: {title}')
    plt.title(title)

    scalarmappaple = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalarmappaple.set_array(acts)
    plt.colorbar(scalarmappaple)
    plt.show()
    return fig

# Cell
@delegates(LineCollection.__init__)
def cam_batch_plot_multi_fig(batch, model, layer=5, func_cam=cam_acts, reduction='mean',force_scale=True, scale_range=(0, 1),                        cmap="Spectral_r", linewidth=4, linestyles='solid', alpha=1.0, transform=None, scatter=False, i2o=noop,
                        figsize=(13, 4), confidence=None, savefig=None, **kwargs):
    'Compute CAM using `func_cam` function, and plot a batch of colored time series `tseries`. The colors correspond to the        scaled CAM values. Each time series is plotted on a separate figure'

    # batch use-cases
    # batch can be either:
    #   - a real batch meaning a tuple of (a list of tseries, a list of y)
    #   - a list of tuples (tseries, y) that we build from a dataset (a dataset item is a tuple of (tseries, y)
    #   - a single dataset item meaning a a tuple of (tseries, y)

    # print(f'Confidence: {confidence}')
    if not isinstance(batch, list):
        if len(batch[0].shape)==3:   # a real batch meaning a tuple of (a list of tseries, a list of y)
            b = itemize(batch)
        elif len(batch[0].shape)==2: # a single dataset item meaning a a tuple of (tseries, y)m
            b = [batch]
    else:
        b = batch # a list of tuples (tseries, y) that we build from a dataset (a dataset item is a tuple of (tseries, y)ch
    n_samples = len(b)

    # b = _listify(b)
    tseries_list, acts_list, y_list, ts_min_max, acts_min_max = CAM_batch_compute(b, model, layer=layer, func_cam=func_cam,                                              reduction=reduction, force_scale=force_scale, transform=transform, scale_range=scale_range, **kwargs)
    levels = 254
    cmap = plt.get_cmap(cmap, lut=levels) #seismic
    if force_scale==True:
        colors = cmap(np.linspace(scale_range[0], scale_range[1], levels))
        norm = BoundaryNorm(np.linspace(scale_range[0], scale_range[1], levels+1), len(colors))
    else:
        colors = cmap(np.linspace(acts_min_max[0], acts_min_max[1], levels))
        norm = BoundaryNorm(np.linspace(acts_min_max[0], acts_min_max[1], levels+1), len(colors))


    # fig, axs = get_grid(4, return_fig=True, figsize=(10, 8))
    # Plot activation `acts` (superimposed on the original time series `tseries)
    lc_list=[]
    title_list=[]
    idx = 1
    for tseries, acts, y in zip(tseries_list, acts_list, y_list):
        t = np.arange(tseries.shape[-1])
        if scatter==True:
            plt.figure(figsize=figsize)
            plt.subplot(1, n_samples, idx)
            plt.scatter(t, tseries, cmap=cmap, c = acts)
            title = i2o(y)
            if not hasattr(func_cam, 'name'): func_cam.name = str(func_cam)
            title =  f'[{title}] - {func_cam.name} - {reduction}'
            if confidence!=None: title = f'{title}\n {confidence[idx-1]}'
            plt.xlim(t.min(), t.max())
            plt.ylim(ts_min_max[0]*1.2, ts_min_max[1]*1.2)
            plt.title(title)
            scalarmappaple = cm.ScalarMappable(norm=norm, cmap=cmap)
            scalarmappaple.set_array(acts)
            plt.colorbar(scalarmappaple)
            plt.show()
            idx += 1
        else:
            points = np.array([t, tseries]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, linestyles=linestyles, alpha=alpha,                                    **kwargs )
            lc.set_array(acts)
            lc.set_linewidth(linewidth)
            lc_list.append(lc)

            title = i2o(y)
            if not hasattr(func_cam, 'name'): func_cam.name = str(func_cam)
            title =  f'[{title}] - {func_cam.name} - {reduction}'
            if confidence!=None: title = f'{title}\n {confidence[idx-1]}'
            title_list.append(title)
            idx += 1

    # Grid
    nrows = int(math.sqrt(n_samples))
    ncols = int(np.ceil(n_samples/nrows))
    fig,axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i<n_samples else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n_samples]
    fig.tight_layout(pad=3.0)

    idx = 0
    if scatter==False:
        for tseries, acts, lc, title in zip(tseries_list, acts_list, lc_list, title_list):
            im = axs[idx].add_collection(lc)
            axs[idx].set_xlim([t.min(), t.max()])
            axs[idx].set_ylim([ts_min_max[0]*1.2, ts_min_max[1]*1.2])
            axs[idx].set_title(title)
            scalarmappaple = cm.ScalarMappable(norm=norm, cmap=cmap)
            scalarmappaple.set_array(acts)
            fig.colorbar(im, ax=axs[idx])
            idx += 1
        plt.show()

    if savefig!=None: plt.savefig(savefig)

    return fig

# Cell
# Example of i2o() function
# Converting CategoryTensor label into the human-readable label
lbl_dict = dict([
    (0, 'Gun'),
    (1, 'Point')]
)
def i2o(y):
    return lbl_dict.__getitem__(y.data.item())
    # return lbl_dict.__getitem__(int(dls.tfms[1][1].decodes(y)))
