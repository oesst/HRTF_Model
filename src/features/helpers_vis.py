import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import src.features.filters as filters
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import skewnorm, norm
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Define some colors
C0 = [31 / 255, 119 / 255, 180 / 255]
C1 = [255 / 255, 127 / 255, 14 / 255]
C2 = [44 / 255, 160 / 255, 44 / 255]
C3 = [214 / 255, 39 / 255, 40 / 255]


MY_COLORS = [C0, C1, C2, C3]


class MLERegression:
    def __init__(self, x, y):

        self.x = np.squeeze(x[:, 1]).reshape(-1, 1)
        self.y = y.reshape(-1, 1)

        self.rr = 1

        # MLE Fit with Skew-Normal Residuals
        def negative_log_likelihood(params, x, y):
            beta0, beta1, alpha, sigma = params
            predicted = beta0 + beta1 * x
            residuals = y - predicted
            nll = -np.sum(skewnorm.logpdf(residuals, a=alpha, loc=0, scale=sigma))
            return nll

        def MLE_Norm(parameters, x, y):
            const, beta, std_dev = parameters
            pred = const + beta*x

            LL = np.sum(norm.logpdf(y, pred, std_dev))
            neg_LL = -1*LL
            return neg_LL

        initial_params = [1, 1, 1, 1]  # beta0, beta1, alpha, sigma
        result_mle = minimize(negative_log_likelihood, initial_params, args=(self.x, self.y), method='L-BFGS-B',
                              bounds=[(None, None), (None, None), (None, None), (1e-6, None)])
        self.beta0_mle, self.beta1_mle, alpha_mle, sigma_mle = result_mle.x

    def get_fitted_line(self):
        return [self.x, self.beta0_mle + self.beta1_mle * self.x]
        # return [self.x, self.lr_model.predict(self.x)]

    def get_coefficients(self):
        # gain, bias
        return self.beta1_mle, self.beta0_mle

    def get_score(self, x=0, y=0):
        if x == 0 or y == 0:
            return self.rr
        else:
            return 1

    def print_coefficients(self):
        gain = self.beta1_mle
        bias = self.beta0_mle
        score = self.rr

        print(
            "Gain: {0:1.2f}, Bias: {1:1.2f}, , r^2: {2:1.2f}".format(
                gain, bias, score
            )
        )
        return ("Gain: {0:1.2f},\nBias: {1:1.2f},\n" + r"$r^2$: {2:1.2f}").format(
            gain, bias, score
        )


class LinearReg:
    def __init__(self, x, y):
        from sklearn.linear_model import LinearRegression

        self.lr_model = LinearRegression()

        if len(x.shape) == 1:
            self.x = x.reshape(-1, 1)
        else:
            self.x = np.squeeze(x[:, 1]).reshape(-1, 1)
        self.y = y.reshape(-1, 1)

        self.lr_model.fit(self.x, self.y)

        self.rr = self.lr_model.score(self.x, self.y)

    def get_fitted_line(self):
        return [self.x, self.lr_model.predict(self.x)]

    def get_coefficients(self):
        # gain, bias
        return self.lr_model.coef_[0, 0], self.lr_model.intercept_[0]

    def get_score(self, x=0, y=0):
        if x == 0 or y == 0:
            return self.rr
        else:
            return self.lr_model.score(x, y)

    def print_coefficients(self):
        print(
            "Gain: {0:1.2f}, Bias: {1:1.2f}, , r^2: {2:1.2f}".format(
                self.lr_model.coef_[0, 0], self.lr_model.intercept_[0], self.rr
            )
        )
        return ("Gain: {0:1.2f},\nBias: {1:1.2f},\n" + r"$r^2$: {2:1.2f}").format(
            self.lr_model.coef_[0, 0], self.lr_model.intercept_[0], self.rr
        )

def get_localization_coefficients_score(x_test, y_test):
    x_test = np.reshape(x_test, (x_test.shape[0] * x_test.shape[1], 2))
    y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1]))

    lr_model = LinearReg(x_test, y_test)
    gain, bias = lr_model.get_coefficients()
    return gain, bias, lr_model.get_score()


def removeOutliers(x, outlierConstant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return np.array(resultList)


def scale_v(x_test, y_test, n_elevations):
    a = x_test[:, :, 1] / n_elevations
    a = a * (n_elevations - 1) * 5.625 - 45
    x_test[:, :, 1] = a

    a = y_test[:, :] / n_elevations
    a = a * (n_elevations - 1) * 5.625 - 45
    y_test[:, :] = a

    return x_test, y_test


def plot_localization_result(
    x_test,
    y_test,
    ax,
    sound_files,
    scale_values=False,
    linear_reg=True,
    disp_values=False,
    scatter_data=True,
    reg_color="",
):
    n_sound_types = len(sound_files)
    if scale_values:
        x_test, y_test = scale_v(x_test, y_test, x_test.shape[0])

    x_test = np.reshape(x_test, (x_test.shape[0] * x_test.shape[1], 2))
    y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1]))

    ax.plot(
        np.arange(np.ceil(np.min(x_test[:, 1])), np.ceil(np.max(x_test[:, 1]))),
        np.arange(np.ceil(np.min(x_test[:, 1])), np.ceil(np.max(x_test[:, 1]))),
        color="grey",
        linestyle="--",
        alpha=0.7,
        label="_nolegend_",
        linewidth=2,
    )

    # error_mse = 0
    for i in range(0, n_sound_types):
        # get the data points, NOT the sound file type
        x = x_test[x_test[:, 0] == i, 1]
        y = y_test[x_test[:, 0] == i]

        if scatter_data:
            ax.scatter(
                x,
                y,
                s=(n_sound_types / (i + 1)) * len(sound_files),
                alpha=0.85,
                label=sound_files[i].name.split(".")[0],
            )
        # error_mse += ((y - x) ** 2).mean(axis=0)

    # error_mse /= n_sound_types

    if linear_reg:
        lr_model = LinearReg(x_test, y_test)
        [x, y] = lr_model.get_fitted_line()

        if scatter_data:
            ax.plot(x, y, color="black")
        else:
            if len(reg_color) > 0:
                ax.plot(x, y, alpha=0.2, linewidth=2, color=reg_color)
            else:
                ax.plot(x, y, alpha=0.2, linewidth=2)

        if disp_values:
            text_str = lr_model.print_coefficients()
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle="round", facecolor="white", alpha=0.8)

            # place a text box in upper left in axes coords
            ax.text(0.70, 0.30, text_str, transform=ax.transAxes, verticalalignment='top', bbox=props)

        # lr_model = MLERegression(x_test, y_test)
        # [x, y] = lr_model.get_fitted_line()

        # ax.plot(x, y, color="black", linestyle="--", label="MLE")
        # text_str = lr_model.print_coefficients()
        #     # these are matplotlib.patch.Patch properties
        # props = dict(boxstyle="round", facecolor="white", alpha=0.8)

        # # place a text box in upper left in axes coords
        # ax.text(0.45, 0.95, text_str, transform=ax.transAxes, verticalalignment="top", bbox=props)

    # ax.set_xlabel('True Elevation')
    # ax.set_ylabel('Elevation based on WN')

    return ax


def set_layout(drawing_size=25, regular_seaborn=False, box_frame=True):
    rc = {'figure.figsize': (10, 5),
          'axes.facecolor': 'white',
          'axes.grid': True,
          'grid.color': '.9',
          'grid.linestyle': ':',
          'font.family': 'Times New Roman',
          'font.size': 20,
          'figure.dpi': 100,
          'savefig.dpi': 400,
          'image.cmap': 'viridis',
          'axes.spines.right': False,
          'axes.spines.top': False, }

    plt.rcParams.update(rc)
    # if regular_seaborn:
    #     import seaborn as sns

    #     sns.set(color_codes=True)
    #     plt.style.use("seaborn-whitegrid")
    #     # sns.set_context("poster")
    #     # sns.set(font="Arial")

    #     # plt.style.use('seaborn')
    #     # sns.set_style("ticks")
    # import seaborn as sns

    # # 21 defines the number of sound types
    # # sns.set_palette(sns.color_palette("husl", 21))
    # mpl.rcParams["grid.linestyle"] = ":"

    # mpl.rcParams["font.size"] = drawing_size
    # mpl.rcParams["font.style"] = "normal"
    # # mpl.rcParams['font.family'] = ['Symbol']

    # mpl.rcParams["figure.titlesize"] = int(drawing_size * 1.3)

    # mpl.rcParams["lines.linewidth"] = int(drawing_size / 5)

    # mpl.rcParams["axes.labelsize"] = drawing_size
    # mpl.rcParams["axes.titlesize"] = int(drawing_size * 1.3)
    # mpl.rcParams["xtick.labelsize"] = int(drawing_size * 1)
    # mpl.rcParams["ytick.labelsize"] = int(drawing_size * 1)

    # if box_frame:
    #     mpl.rcParams["legend.fancybox"] = True
    #     mpl.rcParams["legend.fontsize"] = int(drawing_size * 1)
    #     mpl.rcParams["legend.frameon"] = True
    #     mpl.rcParams["legend.framealpha"] = 0.5
    # else:
    #     mpl.rcParams["legend.fancybox"] = False
    #     mpl.rcParams["legend.fontsize"] = int(drawing_size * 1)
    #     mpl.rcParams["legend.frameon"] = False
    # mpl.rcParams["legend.facecolor"] = "inherit"
    # mpl.rcParams["legend.edgecolor"] = "0.8"

    # mpl.rcParams["figure.figsize"] = [20.0, 10.0]
    # mpl.rcParams["figure.dpi"] = 100
    # mpl.rcParams["savefig.dpi"] = 400
    # mpl.rcParams["image.cmap"] = "viridis"


def mesh_plot(data):
    # fig is the figure handler
    # data is a 2d array
    # returns the axis of the figure

    # define x and y axis
    x = np.linspace(np.min(data[0, :]), np.max(data[0, :]), data.shape[0])
    y = np.linspace(np.min(data[1, :]), np.max(data[1, :]), data.shape[1])
    # x = np.linspace(0, 1, data.shape[0])
    # y = np.linspace(0, 1, data.shape[1])
    x, y = np.meshgrid(x, y)
    # create figure with 3d subplot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # plot data with nice colors
    ax.plot_surface(x, y, data.T, rstride=1, cstride=1, cmap=cm.jet, shade=True, linewidth=20)

    # c = ax.contour(x , y , tmp.T, colors='black', zorder=0)
    # # Thicken the zero contour.
    # zc = c.collections[:]
    # plt.setp(zc, linewidth=4)
    return ax


def set_correct_axis_3D(ax, cutoff_freq_low, cutoff_freq_high):
    ax.set_ylabel("Frequencies (kHz)")
    ticks, _ = plt.yticks()
    ax.set_yticklabels([str(int(k / 1000)) for k in np.linspace(cutoff_freq_low, cutoff_freq_high, len(ticks) + 1)])

    ax.set_zlabel("SPL (au)")
    ax.set_xlabel("Elevation")
    ticks, _ = plt.xticks()
    ax.set_xticklabels([str(int(k)) for k in np.linspace(-45, 90, len(ticks))])


def set_correct_axis_2D(ax, cutoff_freq_low, cutoff_freq_high):
    ax.set_xlabel("Frequencies (kHz)")
    ticks, _ = plt.xticks()
    ax.set_xticklabels([str(int(k / 1000)) for k in np.linspace(cutoff_freq_low, cutoff_freq_high, len(ticks) + 1)])

    ax.set_ylabel("Elevation")
    ticks, _ = plt.yticks()
    ax.set_yticklabels([str(int(k)) for k in np.linspace(-45, 90, len(ticks))])


def plot_elevation_map(data, sounds, figsize=(25, 5)):
    fig = plt.figure(figsize=figsize)

    for i in range(0, data.shape[0], 1):
        ax = fig.add_subplot(1, data.shape[0], 1 + i)
        ax.set_title(sounds[i].split("/")[1].split("_")[0].split(".")[0])
        data_revised_i = np.squeeze(data[i, :, :])

        c = ax.pcolormesh(data_revised_i[:, :])

        plt.colorbar(c)


def set_axis(ax, n_elevations=25, label=False):
    if label:
        ax.set_xlabel("True Elevation [deg]")
    t = np.zeros(6)

    if n_elevations == 25:
        t[0] = -55
        t[1] = -45
        t[2] = 0
        t[3] = 45
        t[4] = 90
        t[5] = 100
        ax.set_xticks(t[1:-1])
        ax.set_yticks(t[1:-1])
    elif n_elevations == 50:
        t = np.zeros(9)

        t[0] = -55
        t[1] = -45 + 45 * 0
        t[2] = -45 + 45 * 1
        t[3] = -45 + 45 * 2
        t[4] = -45 + 45 * 3
        t[5] = -45 + 45 * 4
        t[6] = -45 + 45 * 5
        t[7] = -45 + 45 * 6
        t[8] = -45 + 45 * 7
        ax.set_xticks(t[1:-1])
        ax.set_yticks(t[1:-1])
    else:
        t[0] = -55
        t[1] = -45
        t[2] = -45 + ((n_elevations * 5.625) / 2.5) * 1
        t[3] = -45 + ((n_elevations * 5.625) / 2.5) * 2
        t[4] = -45 + ((n_elevations * 5.625) / 2.5) * 3
        t[5] = -45 + ((n_elevations * 5.625) / 2.5) * 4
        ax.set_xticks(t[1:-1])
        ax.set_yticks(t[1:-1])
    return ax


def plot_localization_nice(ax, data_frame):
    # ax.plot(x, x, "--", color="grey", alpha=0.7)
    # ax = sns.regplot(
    #     x="line_number",
    #     y="user_estimate",
    #     data=data_frame,
    #     x_estimator=np.mean,
    #     x_ci="sd",
    #     color="#525252",
    #     line_kws={"linewidth": 3},
    #     ax=ax,
    #     scatter_kws={"zorder": 10},
    # )

    ax.set_ylabel("Estimated Elevation [deg]")
    ax.set_xlabel("True Elevation [deg]")
    ax.tick_params(labelleft=True)
    for i, I in enumerate(ax.get_lines()):
        # get the horizontal lines for std
        # this is a hack.
        if i > 93 and i < len(ax.get_lines()) - 1:
            I.set_linewidth(0.5)
    ax.set_aspect("equal", adjustable="box")

    # calculate the linear regression gain separately for each participant
    lr_results = np.zeros((len(data_frame.user_id.unique()), 4))
    for i_id, id in enumerate(data_frame.user_id.unique()):
        df__ = data_frame[data_frame.user_id == id]
        perc = df__.user_estimate
        outp = df__.line_number
        lr = LinearReg(outp.values, perc.values)
        g, b = lr.get_coefficients()
        s = lr.get_score()
        lr_results[i_id, :] = g, b, s, df__.shape[0]

        # calulate weighted mean of the gain and bias

    print()
    x = np.sort(data_frame.line_number.unique())

    print(lr_results.shape)
    mean_gain = np.average(lr_results[:, 0], weights=lr_results[:, 3])
    mean_bias = np.average(lr_results[:, 1], weights=lr_results[:, 3])
    mean_score = np.average(lr_results[:, 2], weights=lr_results[:, 3])

    ax.plot(x, x * mean_gain + mean_bias, color='black', alpha=1.0, linewidth=3)
    r_squre = r"$r^2$"
    average = r"$\varnothing$"
    text_str = f'{average}Gain: {mean_gain:1.2f}\n{average}Bias : {mean_bias:1.2f}\n {average}{r_squre}: {mean_score:1.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.70, 0.30, text_str, transform=ax.transAxes, verticalalignment='top', bbox=props)

    # calculate confidence intervals of the gain
    gain_slower_bound = np.percentile(lr_results[:, 0], 5)
    gains_upper_bound = np.percentile(lr_results[:, 0], 95)

    # calculate the confidence intervals of the bias
    bias_lower_bound = np.percentile(lr_results[:, 1], 5)
    bias_upper_bound = np.percentile(lr_results[:, 1], 95)

    x = np.sort(data_frame.line_number.unique())

    # plot the confidence intervals (filled area)
    x = x.squeeze()

    # order the x values and y values
    x = x[np.argsort(x)]

    # calcualte the average gain
    mean_gain = lr_results[:, 0].mean()
    mean_bias = lr_results[:, 1].mean()

    # ax.fill_between(x, mean_gain * x + lower_bound + mean_bias, mean_gain * x + upper_bound + mean_bias, color=color, alpha=0.2)

    ax.fill_between(x, x * gain_slower_bound + mean_bias, x * gains_upper_bound + mean_bias, color='black', alpha=0.2)
    ax.fill_between(x, x * gains_upper_bound + mean_bias, x * gain_slower_bound + mean_bias, color='black', alpha=0.2)

    # ax.fill_between(x, x * mean_gain + bias_lower_bound, x * mean_gain + bias_upper_bound, color='black', alpha=0.2)
    # ax.fill_between(x, x * mean_gain + bias_upper_bound, x * mean_gain + bias_lower_bound, color='black', alpha=0.2)

    ax.plot(x, x * mean_gain + mean_bias, color='black', linestyle="-", linewidth=2, label="Mean")

    return ax


# def set_axis_all_elevations(ax, label=False):
#     if label:
#         ax.set_xlabel('True Elevation [deg]')
#
#
#
#     return ax


class ERBFormatter(mpl.ticker.EngFormatter):
    """
    Axis formatter for gammatone filterbank analysis. This formatter calculates
    the ERB spaced frequencies used for analysis, and renders them similarly to
    the engineering axis formatter.
    The scale is changed so that `[0, 1]` corresponds to ERB spaced frequencies
    from ``high_freq`` to ``low_freq`` (note the reversal). It should be used
    with ``imshow`` where the ``extent`` argument is ``[a, b, 1, 0]`` (again,
    note the inversion).
    """

    def __init__(self, low_freq, high_freq, *args, **kwargs):
        """
        Creates a new :class ERBFormatter: for use with ``matplotlib`` plots.
        Note that this class does not supply the ``units`` or ``places``
        arguments; typically these would be ``'Hz'`` and ``0``.
        :param low_freq: the low end of the gammatone filterbank frequency range
        :param high_freq: the high end of the gammatone filterbank frequency
          range
        """
        self.low_freq = high_freq
        self.high_freq = low_freq
        super().__init__(*args, **kwargs)

    def _erb_axis_scale(self, fraction):
        return filters.erb_point(self.low_freq, self.high_freq, fraction)

    def __call__(self, val, pos=None):
        newval = self._erb_axis_scale(val)
        return super().__call__(newval, pos)


def make_double_side_plot(fig, ax, df_model, df_real, lr_results_model, lr_results_real, lmm_results=None):
    """
    Create a side plot for the given parameters. Mainly for model and real data comparison.
    """
    # ax2 = fig.add_axes([ax.get_position().x1 - 0.00, ax.get_position().y0, 0.03, ax.get_position().height])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)

    #
    ax, ax_violin, mean_bias = make_side_plot(fig, ax, df_real, lr_results_real, lmm_results=lmm_results, color='C0', side="right")
    ax, ax_violin, mean_bias = make_side_plot(fig, ax, df_model, lr_results_model, lmm_results=lmm_results, color='C1', side="left", mean_bias=mean_bias)

    # calculate plot units
    left_plot = 53 + 53  # range of the plot
    right_plot = 1 + 1  # range of the plot
    ratio = (left_plot / right_plot)

    # calculate mean gain and bias
    gain_values_real = lr_results_real[:, 0] + mean_bias / ratio
    gain_values_model = lr_results_model[:, 0] + mean_bias / ratio

    print(gain_values_real.shape, gain_values_model.shape)
    # do a scatter plot of the real and model data and connect them with a line
    ax_violin.scatter(np.zeros(lr_results_real.shape[0]) + 0.5 + np.random.uniform(-0.05, 0.05, size=lr_results_real.shape[0]), gain_values_real, color='C0', alpha=1.0, s=25)
    ax_violin.scatter(np.zeros(lr_results_model.shape[0]) - 0.5 + np.random.uniform(-0.05, 0.05, size=lr_results_model.shape[0]), gain_values_model, color='C1', alpha=1.0, s=25)
    # connect the real and model data with a line
    gain_difference = np.abs(gain_values_real - gain_values_model)
    for i in range(lr_results_real.shape[0]):
        ax_violin.plot([0.5 + np.random.uniform(-0.05, 0.05), -0.5 + np.random.uniform(-0.05, 0.05)], [gain_values_real[i], gain_values_model[i]], color='grey', alpha=0.8, linewidth=1)

    # show that the difference is significant and ploit a star
    t_statistic, p_value = stats.ttest_rel(gain_values_real, gain_values_model)
    print(f"T-statistic: {t_statistic}, P-value: {p_value}")
    if p_value < 0.001:
        # plot a star
        ax_violin.text(0, 1.2, '***', fontsize=20, ha='center', va='center', color='black')
        # plot a line
        ax_violin.plot([-0.5, 0.5], [1.2, 1.2], color='black', linewidth=2.0)
    elif p_value < 0.01:
        # plot two stars
        ax_violin.text(0, 1.1, '**', fontsize=20, ha='center', va='center', color='black')
        # plot a line
        ax_violin.plot([-0.5, 0.5], [1.1, 1.1], color='black', linewidth=2.0,)

    elif p_value < 0.05:
        # plot three stars
        ax_violin.text(0, 1.2, '*', fontsize=20, ha='center', va='center', color='black')
        # plot a line
        ax_violin.plot([-0.5, 0.5], [1.1, 1.2], color='black', linewidth=2.0)
    return ax, ax_violin, mean_bias


def make_double_side_plot(fig, ax, df_model, df_real, lr_results_model, lr_results_real, lmm_results=None):
    """
    Create a side plot for the given parameters. Mainly for model and real data comparison.
    """
    # ax2 = fig.add_axes([ax.get_position().x1 - 0.00, ax.get_position().y0, 0.03, ax.get_position().height])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)

    #
    ax, ax_violin, mean_bias = make_side_plot(fig, ax, df_real, lr_results_real, lmm_results=lmm_results, color='C0', side="right")
    ax, ax_violin, mean_bias = make_side_plot(fig, ax, df_model, lr_results_model, lmm_results=lmm_results, color='C1', side="left", mean_bias=mean_bias)

    # calculate plot units
    left_plot = 53 + 53  # range of the plot
    right_plot = 1 + 1  # range of the plot
    ratio = (left_plot / right_plot)

    # calculate mean gain and bias
    gain_values_real = lr_results_real[:, 0] + mean_bias / ratio
    gain_values_model = lr_results_model[:, 0] + mean_bias / ratio

    # # save the gain values for later use in a csv file
    # gain_values_df = pd.DataFrame({
    #     'gain_values_real': gain_values_real,
    #     'gain_values_model': gain_values_model
    # })
    # if df_real.ear_condition.unique()[0] == 'monaural' and df_real.sound_group.unique()[0] == 'rippled':
    #     gain_values_df.to_csv('gain_values.csv', index=False)

    # do a scatter plot of the real and model data and connect them with a line
    ax_violin.scatter(np.zeros(lr_results_real.shape[0]) + 0.5 + np.random.uniform(-0.05, 0.05, size=lr_results_real.shape[0]), gain_values_real, color='C0', alpha=1.0, s=25)
    ax_violin.scatter(np.zeros(lr_results_model.shape[0]) - 0.5 + np.random.uniform(-0.05, 0.05, size=lr_results_model.shape[0]), gain_values_model, color='C1', alpha=1.0, s=25)
    # connect the real and model data with a line
    gain_difference = np.abs(gain_values_real - gain_values_model)
    for i in range(lr_results_real.shape[0]):
        ax_violin.plot([0.5 + np.random.uniform(-0.05, 0.05), -0.5 + np.random.uniform(-0.05, 0.05)], [gain_values_real[i], gain_values_model[i]], color='grey', alpha=0.8, linewidth=1)

    # show that the difference is significant and ploit a star
    t_statistic, p_value = stats.ttest_rel(lr_results_real[:, 0], lr_results_model[:, 0])
    print(f"T-statistic: {t_statistic}, P-value: {p_value}")
    if p_value < 0.001:
        # plot a star
        ax_violin.text(0, 1.2, '***', fontsize=20, ha='center', va='center', color='black')
        # plot a line
        ax_violin.plot([-0.5, 0.5], [1.2, 1.2], color='black', linewidth=2.0)
    elif p_value < 0.01:
        # plot two stars
        ax_violin.text(0, 1.1, '**', fontsize=20, ha='center', va='center', color='black')
        # plot a line
        ax_violin.plot([-0.5, 0.5], [1.1, 1.1], color='black', linewidth=2.0,)

    elif p_value < 0.05:
        # plot three stars
        ax_violin.text(0, 1.2, '*', fontsize=20, ha='center', va='center', color='black')
        # plot a line
        ax_violin.plot([-0.5, 0.5], [1.1, 1.2], color='black', linewidth=2.0)
    return ax, ax_violin, mean_bias


def make_side_plot(fig, ax, df, lr_results, lmm_results=None, color='C0', side="right", mean_bias=None):
    """
    Create a side plot for the given parameters.
    # """
    # ax2 = fig.add_axes([ax.get_position().x1 - 0.00, ax.get_position().y0, 0.03, ax.get_position().height])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # # make a violin plot of the gain values without using seaborn
    # # ax.violinplot(lr_results[:, 0], vert=False, showmeans=True, showmedians=True, widths=0.5, points=100)

    # Plot the violin plot on the right-hand side
    ax_violin = ax.inset_axes([1.01, 0.0, 0.1, 1.2])  # [left, bottom, width, height]

    # calculate plot units
    left_plot = 53 + 53  # range of the plot
    right_plot = 1 + 1  # range of the plot
    ratio = (left_plot / right_plot)

    # get bias corrected gain values
    # avg_bias = np.average(lr_results[:, 1], weights=lr_results[:, 3])
    if mean_bias is None:
        if lmm_results is not None:
            # use the linear mixed model results to get the mean bias
            mean_bias = np.average(lmm_results.fe_params['Intercept'])
        else:
            # use the linear regression results to get the mean bias
            mean_bias = np.average(lr_results[:, 1], weights=lr_results[:, 3])
    else:
        # use the provided mean bias
        mean_bias = mean_bias

    # calculate mean gain and bias
    gain_values = lr_results[:, 0] + mean_bias / ratio

    # plot the mean in the violin plot
    violin = ax_violin.violinplot(gain_values, positions=[0], showmeans=True, showmedians=False, widths=4.0, showextrema=False, quantiles=[[0.05, 0.95]])

    # make mean lines thicker
    violin['cmeans'].set_linewidth(2.0)
    violin['cmeans'].set_color(color)
    # make quantiles lines thicker
    violin['cquantiles'].set_linewidth(1.0)
    violin['cquantiles'].set_color(color)

    for pc in violin['bodies']:
        pc.set_linewidth(1.0)
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.5)

        if side == "left":
            # get the center
            m = np.mean(pc.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further right than the center
            pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
        else:
            # get the center
            m = np.mean(pc.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further left than the center
            pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)

    # do also a scatter plot of the gain values
    if side == "left":
        ax_violin.scatter(np.zeros(lr_results.shape[0]) - 0.5 + np.random.uniform(-0.05, 0.05, size=lr_results.shape[0]), gain_values, color=color, alpha=1.0, s=20)
    else:
        ax_violin.scatter(np.zeros(lr_results.shape[0]) + 0.5 + np.random.uniform(-0.05, 0.05, size=lr_results.shape[0]), gain_values, color=color, alpha=1.0, s=20)

    # Remove x-axis from violin plot
    ax_violin.set_xticks([])
    # ax_violin.set_xlabel("")
    ax_violin.set_ylim([-1 + mean_bias / ratio, 1.4 - mean_bias / ratio])
    if side == "left":
        ax_violin.set_xlim([-2.0, 2.0])
    else:
        ax_violin.set_xlim([-2.0, 2.0])

    # change the y tick labels by the mean bias
    ax_violin.set_yticks(np.array([0 + mean_bias / ratio, 1]))
    ax_violin.set_yticklabels(['0', '1'])
    ax_violin.spines['left'].set_bounds(0 + mean_bias / ratio, 1)

    # remove y axis lines from the violin plot
    ax_violin.spines['left'].set_visible(True)
    ax_violin.spines['right'].set_visible(False)
    ax_violin.spines['top'].set_visible(False)
    ax_violin.spines['bottom'].set_visible(False)

    # set the right spine length

    # make background transparent
    ax_violin.set_facecolor('none')
    # Set y-ticks on the left spine
    ax_violin.yaxis.set_ticks_position('left')

    # Make ticks point inward (i.e., into the plot)
    ax_violin.tick_params(axis='y', direction='in', length=6)

    # Optional: adjust padding so labels stay left of spine
    ax_violin.tick_params(axis='y', pad=-20)
    # remove grid lines
    ax_violin.grid(False)

    return ax, ax_violin, mean_bias


def plot_linear_mixed_model(df_, ax, color='C0', plot_text=True):
    # use a linear mixed model to fit the data
    model = smf.mixedlm("user_estimate ~ line_number", df_, groups=df_["user_id"], re_formula="~line_number")
    # use many optimization methods to fit the model
    result = model.fit(method=['lbfgs', 'bfgs', 'powell', 'cg', 'nm'], maxiter=1000)
    # result = model.fit(method=['lbfgs'], maxiter=1000)
    # Print summary with more digits after the decimal point
    print(result.summary().as_text())

    for i, group in enumerate(df_["user_id"].unique()):
        subset = df_[df_['user_id'] == group]
        # ax.scatter(subset['line_number'], subset['user_estimate'], label=f'{group} data')

        # Fit separate OLS lines for visual comparison (not part of LMM)
        coef = np.polyfit(subset['line_number'], subset['user_estimate'], 1)
        x_fit = np.linspace(subset['line_number'].min(), subset['line_number'].max(), 100)
        y_fit = coef[0] * x_fit + coef[1]
        # ax.plot(x_fit, y_fit, linestyle='--', alpha=0.6)

    # Plot population-level (fixed effect) line
    # Create the X_pred matrix (100 x 2): intercept + x
    x_pred = np.linspace(df_['line_number'].min(), df_['line_number'].max(), 100)
    X_pred = np.column_stack((np.ones_like(x_pred), x_pred))  # shape (100, 2)

    # Extract just the fixed effects (Intercept and slope)
    fe_params = result.fe_params[['Intercept', 'line_number']].values     # shape (2,)
    vcov = result.cov_params().loc[['Intercept', 'line_number'], ['Intercept', 'line_number']].values  # shape (2, 2)

    # Predicted values
    y_pred = X_pred @ fe_params

    # Standard error of predictions
    se_pred = np.sqrt(np.sum((X_pred @ vcov) * X_pred, axis=1))  # shape (100,)

    # 95% CI band
    y_lower = y_pred - 1.96 * se_pred
    y_upper = y_pred + 1.96 * se_pred

    print("Max CI width:", (y_upper - y_lower).max())
    print("Intercept SE:", result.bse['Intercept'])
    print("Slope SE:", result.bse['line_number'])

    # Plot
    ax.plot(x_pred, y_pred, color=color, label='Population fit', linewidth=3)
    ax.fill_between(x_pred, y_lower, y_upper, color=color, alpha=0.3, label='95% CI')

    # plot mean user estimates from model
    # Unique values of the predictor
    x_unique = np.sort(df_['line_number'].unique())

    # Access model components
    fe = result.fe_params

    x_levels = sorted(df_['line_number'].unique())
    y_means = []
    y_stds = []
    y_sems = []

    for x in x_levels:
        # Get subset of data where this line_number was used
        subset = df_[df_['line_number'] == x]

        preds = []
        for i, row in subset.iterrows():
            g = row['user_id']
            ri = result.random_effects[g]

            # Handle group key naming
            intercept_g = fe['Intercept'] + ri['Group']
            slope_g = fe['line_number'] + ri['line_number']

            preds.append(intercept_g + slope_g * x)

        preds = np.array(preds)
        n = len(preds)
        y_means.append(np.mean(preds))
        y_stds.append(np.std(preds, ddof=1))  # Sample standard deviation
        y_sems.append(np.std(preds, ddof=1) / np.sqrt(n))  # SEM

    # calculate mean gain and bias
    mean_gain = np.average(result.fe_params['line_number'])
    mean_bias = np.average(result.fe_params['Intercept'])

    # Convert to arrays for plotting
    y_means = np.array(y_means)
    y_stds = np.array(y_stds)
    y_sems = np.array(y_sems)
    # Plot the mean predicted values as black dots
    # ax.scatter(x_levels, y_means, color='black', s=60, zorder=3, label='Model mean at x')
    ax.errorbar(x_levels, y_means, yerr=y_stds, fmt='o', color=color, label='Model mean at x', markersize=8, capsize=0)

    ax.set_ylabel('Indicated Location [deg]')
    ax.set_xlabel('Sound Location [deg]')
    x = np.sort(df_.line_number.unique())

    # extend the x-axis to the right
    x = list(x) + [45 + 2 * 11.25, 45 + 3 * 11.25]
    ax.plot(x, x + mean_bias, '--', color='grey', linewidth=1.0, alpha=0.7)
    ax.axhline(mean_bias, color='grey', linewidth=1.0, linestyle='--', alpha=0.7)

    x_min = -53
    x_max = 53
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)

    r_squre = r"$r^2$"
    average = r"$\varnothing$"
    # text_str = f'{average}Gain: {mean_gain:1.2f}\n{average}Bias : {mean_bias:1.2f}\n{average}{r_squre}: {mean_score:1.2f}'
    # text_str = f'{average}Gain: {mean_gain:1.2f}\n{average}Bias : {mean_bias:1.2f}'
    text_str = f'Gain: {mean_gain:1.2f}\nBias : {mean_bias:1.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    if plot_text:
        ax.text(0.60, 0.30, text_str, transform=ax.transAxes, verticalalignment='top', bbox=props)

    return ax, result
