"""
group_plots.py — regenerate ONLY the group-level figures
(group_lollipop_raincloud.png and group_paired_bar_grid.png)
without rerunning process_participant() / the full EEG pipeline.

Why this works without reprocessing:
build_group_trials_table / build_group_summary_table / build_group_effects_table
only read the per-pulse-feature CSVs and spindle-summary CSVs that
process_participant() already wrote to OUTPUT_DIR the last time you ran
analysis.py. They never touch the raw .fif files. So as long as those CSVs
already exist on disk, this script can rebuild the group tables and redraw
the group plots in seconds.

Usage:
    python group_plots.py
    python group_plots.py --participants 03 06 08
"""

import argparse
from pathlib import Path

# Importing analysis.py only runs its module-level code (imports, constant
# definitions, function definitions). Its actual pipeline is guarded behind
# `if __name__ == '__main__':`, so importing it here does NOT reprocess
# any participants or reload any raw EEG data.
import analysis_stats as A


def main():
    parser = argparse.ArgumentParser(description='Regenerate group-level plots only')
    parser.add_argument('--participants', nargs='+', default=None,
                         help='Defaults to A.PARTICIPANTS from analysis.py')
    parser.add_argument('--output-dir', default=None,
                         help='Defaults to A.OUTPUT_DIR from analysis.py')
    args = parser.parse_args()

    participants = args.participants if args.participants else A.PARTICIPANTS
    output_dir = Path(args.output_dir if args.output_dir else A.OUTPUT_DIR)

    print(f'Rebuilding group tables for {len(participants)} participant(s): {participants}')

    trials_df  = A.build_group_trials_table(participants, str(output_dir))
    summary_df = A.build_group_summary_table(trials_df)
    effects_df = A.build_group_effects_table(participants, str(output_dir))

    if trials_df.empty or summary_df.empty:
        print('No trial/summary data found — check that per-pulse CSVs exist in '
              f'{output_dir}/<participant>/ from a previous analysis.py run.')
        return

    # --- example subjects for the raincloud panels -------------------------
    # NOTE: this is the fixed version of the buggy call in analysis.py.
    # plot_group_effect_lollipop_raincloud() wants a plain list of subject
    # IDs (e.g. ['03', '06']), NOT (subject, session) tuples — passing
    # tuples is exactly what caused your
    #   ValueError: Lengths of operands do not match: 632 != 2
    # because pandas tried to compare the whole 'subject' column against a
    # 2-element tuple instead of a single subject string.
    EXAMPLE_SUBJECT_IDS = ['03', '06']
    available = [s for s in EXAMPLE_SUBJECT_IDS if s in summary_df['subject'].values]
    if len(available) < len(EXAMPLE_SUBJECT_IDS):
        missing = set(EXAMPLE_SUBJECT_IDS) - set(available)
        print(f'  Warning: no trial data for example subject(s) {sorted(missing)} — skipping')

    A.plot_group_effect_lollipop_raincloud(
        summary_df, trials_df, available,
        output_dir / 'group_lollipop_raincloud.png',
    )

    measures = [
        ('sigma_power', 'Sigma power'), ('delta_power', 'Delta power'),
        ('spindle_rate', 'Spindle rate'), ('spindle_amplitude', 'Amplitude'),
        ('spindle_duration', 'Duration'), ('spindle_frequency', 'Frequency'),
    ]
    A.plot_group_paired_bar_grid(
        effects_df, measures, output_dir / 'group_paired_bar_grid.png',
    )

    print('Done.')


if __name__ == '__main__':
    main()
