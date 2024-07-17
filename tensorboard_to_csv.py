from tbparse import SummaryReader
import pandas as pd


def wall_time_to_relative_time(df):
    # wall_time to actual time
    df['wall_time'] = pd.to_datetime(df['wall_time'], unit='s')

    # time to relative time
    df['wall_time'] = df['wall_time'] - df['wall_time'][df['step'] == 1].values[0]
    df['wall_time'] = df['wall_time'].dt.total_seconds() // 60

def save_to_csv(df, output_file):
    df.to_csv(output_file, index=False)


log_dir = "/app/logs/"
reader = SummaryReader(log_dir,extra_columns={'dir_name','wall_time'}, )
df = reader.scalars

# Run on each experiment separately
for experiment in df['dir_name'].unique():
    df_exp = df[df['dir_name'] == experiment]
    wall_time_to_relative_time(df_exp)
    output_file = f"{log_dir}/csvs/{experiment}.csv"
    save_to_csv(df_exp, output_file)
    print (f"Saved {output_file}")
    

