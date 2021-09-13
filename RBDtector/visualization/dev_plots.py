import matplotlib.pyplot as plt
import os

def dev_plots(df, output_path=None):
    print(df.info())
    df = df.copy()
    start_date = df.index[0].date()
    end_date = df.index[-1].date()

    # df = df.iloc[(df.index.size // 2) + (df.index.size // 7):-(
    #             (df.index.size // 5) + (df.index.size // 9))]  # -(df.index.size//6)
    # df = df.between_time(
    #     # start_time=datetime.datetime.combine(start_date, datetime.time(22, 00)),
    #     start_time=pd.to_datetime('22:00').time(),
    #     # end_time=datetime.datetime.combine(start_date, datetime.time(23, 00)))
    #     end_time=pd.to_datetime('23:00').time())
    signal_type = 'EMG'
    # df = df.loc[df[signal_type + '_phasic_miniepochs']]
    # plt.fill_between(df.index.values,
    #                  df[signal_type + '_human_phasic'] * (-1000),
    #                  df[signal_type + '_human_phasic'] * 1000,
    #                  facecolor='royalblue', label=signal_type + "_human_phasic", alpha=0.7
    #                  )
    #
    # plt.plot(df.index.values, df[signal_type], c='#313133', label=signal_type + ' filtered', alpha=0.7, zorder=4)
    fig, ax = plt.subplots()
    # REM PHASES
    ax.fill_between(df.index.values, df['is_REM'] * (-1000), df['is_REM'] * 1000,
                    facecolor='lightblue', label="is_REM", alpha=0.7)

    ax.fill_between(df.index.values, df['is_artifact'] * (-550),
                    df['is_artifact'] * 550,
                    facecolor='lightpink', label="Arousal", alpha=0.7)
    ax.fill_between(df.index.values, df['artifact_free_rem_sleep_miniepoch'] * (-750),
                    df['artifact_free_rem_sleep_miniepoch'] * 750,
                    facecolor='#e1ebe8', label="Artefact-free REM sleep miniepoch", alpha=0.7)
    # ACTIVITIES
    # ax.fill_between(df.index.values, df[signal_type + '_increased_activity']*(-50),
    #                 df[signal_type + '_increased_activity']*50, alpha=0.7, facecolor='yellow',
    #                 label="0.05s contains activity", zorder=4)
    # ax.fill_between(df.index.values, (df[signal_type + '_min_sustained_activity'])*(-25),
    #                 (df[signal_type + '_min_sustained_activity'])*25, alpha=0.7, facecolor='orange',
    #                 label="minimum_sustained_activity (0.1s)", zorder=4)
    # ax.fill_between(df.index.values, (df[signal_type + '_max_tolerable_gaps'])*(-25),
    #                 (df[signal_type + '_max_tolerable_gaps'])*25, alpha=0.7, facecolor='lightcoral',
    #                 label="Gaps > 0.25s between increased activity", zorder=4)
    ax.fill_between(df.index.values, (df[signal_type + '_sustained_activity']) * (-25),
                    (df[signal_type + '_sustained_activity']) * 25, alpha=0.7, facecolor='orange', edgecolor='darkgrey',
                    label="sustained_activity", zorder=6)
    # ax.fill_between(df.index.values, (df[signal_type + '_any_miniepochs']) * (-40),
    #                 (df[signal_type + '_any_miniepochs']) * 40, alpha=0.7, facecolor='pink',
    #                 label="Any activity", zorder=5)
    # ax.plot(df[signal_type + '_any_miniepochs'] * 40, alpha=0.7, c='pink',
    #         label="Any activity miniepoch", zorder=4)
    # ax.fill_between(df.index.values, (df[signal_type + '_tonic']) * (-25),
    #                 (df[signal_type + '_tonic']) * 25, alpha=0.9, facecolor='gold',
    #                 label="Tonic epoch", zorder=4)

    # ax.fill_between(df.index.values, (df[signal_type + '_phasic_miniepochs']) * (-75),
    #                 (df[signal_type + '_phasic_miniepochs']) * 75, alpha=0.7, facecolor='yellow',
    #                 label="Phasic activity", zorder=4)

    # ax.plot(df[signal_type + '_phasic_miniepochs'] * 75, alpha=0.7, c='deeppink',
    #         label="Phasic activity miniepoch", zorder=4)
    # HUMAN RATING OF CHIN EMG
    # ax.fill_between(df.index.values, df[signal_type + '_human_tonic'] * (-1000),
    #                 df[signal_type + '_human_tonic'] * 1000,
    #                 facecolor='aqua', label=signal_type + "_human_tonic")
    # ax.fill_between(df.index.values, df[signal_type + '_human_intermediate'] * (-1000),
    #                 df[signal_type + '_human_intermediate'] * 1000,
    #                 facecolor='deepskyblue', label=signal_type + "_human_intermediate")
    # ax.fill_between(df.index.values, df[signal_type + '_human_phasic'] * (-1000),
    #                 df[signal_type + '_human_phasic'] * 1000,
    #                 facecolor='royalblue', label=signal_type + "_human_phasic", alpha=0.7)
    ax.fill_between(df.index.values, df[signal_type + '_increased_activity'] * (-5),
                    df[signal_type + '_increased_activity'] * 5,
                    facecolor='lightblue', label=signal_type + "_increased_activity", alpha=0.7, zorder=7)
    ax.fill_between(df.index.values, df[signal_type + '_human_artifact'] * (-1000),
                    df[signal_type + '_human_artifact'] * 1000,
                    facecolor='maroon', label=signal_type + "_human_artifact", zorder=10)
    # ax.fill_between(df.index.values, df['miniepoch_contains_artifact'] * (-75),
    #                  df['miniepoch_contains_artifact'] * 75,
    #                  facecolor='#993404', label="miniepoch_contains_artifact", alpha=0.7, zorder=4)
    # SIGNAL CHANNEL
    ax.plot(df.index.values, df[signal_type], c='#313133', label=signal_type, alpha=0.85, zorder=12)
    # ax.plot(df.index.values, df[signal_type], c='deeppink', label=signal_type + ' filtered', alpha=0.85, zorder=4)
    ax.plot(df[signal_type + '_baseline'], c='mediumseagreen', label=signal_type + "_baseline", zorder=4)
    ax.plot(df[signal_type + '_baseline'] * (-1), c='mediumseagreen', zorder=4)
    ax.plot([df.index.values[0], df.index.values[-1]], [0, 0], c='dimgrey')
    ax.scatter(df.index.values, df[signal_type].where(df[signal_type + '_two_times_baseline_and_valid']), s=4,
               c='lime',
               label='Increased activity', zorder=4)
    # ax.plot(df['diffs'] * 10, c='deeppink', label="Diffs", zorder=4)
    # ARTEFACTS
    # ax.fill_between(df.index.values, df['is_artifact'] * (-200), df['is_artifact'] * 200,
    #                  facecolor='#dfc27d', label="is_artifact", alpha=0.7, zorder=3)
    ax.legend(loc='upper left', facecolor='white', framealpha=1)

    if output_path is not None:
        plt.savefig(os.path.join(os.path.dirname(output_path), os.path.basename(output_path) + '_figure.jpg'),
                    dpi=600)
    # plt.plot(resampled_signal_dtseries.index.values, resampled_signal_dtseries.values)
    # plt.legend()
    plt.show()
    ############################################################################################################