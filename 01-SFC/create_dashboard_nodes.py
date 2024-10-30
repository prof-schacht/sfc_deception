# %%

import requests

import pandas as pd
# load from pickle
df_nodes_explanation_deceptive = pd.read_pickle('./data/df_nodes_explanation_deceptive_sis.pkl')
df_nodes_explanation_truthful = pd.read_pickle('./data/df_nodes_explanation_truthful_sis.pkl')

# %%
# %%

def create_combined_dashboard(df, output_file, title):
    """
    Create a single HTML file containing all dashboard elements from the dataframe.
    
    Args:
        df: DataFrame containing layer, hook_point, and node_id columns
    """
    # HTML header with some basic styling
    html_content = """
    <html>
    <head>
        <style>
            .dashboard-container {
                margin: 20px 0;
                padding: 20px 0;
                border-bottom: 1px solid #ccc;
            }
        </style>
    </head>
    <body>
    <h1>"""+title+"""</h1>
    """
    
    # Loop through dataframe rows
    for _, row in df.iterrows():
        # Convert layer number to string and construct sae_id
        layer = str(row['layer'])
        hook_point_map = {
            'resid_post': 'gemmascope-res-16k',
            'mlp_out': 'gemmascope-mlp-16k'
        }
        
        # Skip if hook_point not in map or node_id is 'error'
        if row['hook_point'] not in hook_point_map or row['node_id'] == 'error':
            continue
            
        sae_id = f"{layer}-{hook_point_map[row['hook_point']]}"
        
        # Create iframe HTML
        dashboard_url = f"https://neuronpedia.org/gemma-2-9b/{sae_id}/{row['node_id']}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
        iframe_html = f"""
        <h2>Layer {layer} - {row['hook_point']} - {row['node_id']}</h2>
        <div class="dashboard-container">
            <iframe src="{dashboard_url}" width="1200" height="600" frameborder="0"></iframe>
        </div>
        """
        html_content += iframe_html
    
    # Close HTML tags
    html_content += """
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)

# Usage example:
# create_combined_dashboard(your_dataframe)
# %%
create_combined_dashboard(df_nodes_explanation_deceptive, output_file='combined_dashboard_deceptive.html', title='Deceptive Nodes')
create_combined_dashboard(df_nodes_explanation_truthful, output_file='combined_dashboard_truthful.html', title='Truthful Nodes')
# %%


url = f"https://www.neuronpedia.org/api/feature/gemma-2-9b/29-gemmascope-res-16k/8545"
        
        #gemma-2-9b/29-gemmascope-res-16k/8545
        #gemma-2-9b/29-gemmascope-res-16K/8545
        
headers = {"X-Api-Key": "sk-np-vbkNl0Y3H8cgzyHNvV5b0BDePGtwh7I7lvwCznEB2tg0"}
        
response = requests.get(url, headers=headers)
# %%
print(response.json()['maxActApprox'])
# %%
