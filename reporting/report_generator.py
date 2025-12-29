import pandas as pd
import matplotlib.pyplot as plt
import jinja2
import markdown
from weasyprint import HTML, CSS
from datetime import datetime
import os


# 1. DATA INGESTION & PROCESSING
# In production, this loads your CSV from the DAQ
def load_and_process_data(csv_path):
    # Mocking data for this example
    data = {
        'time': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'pc': [1.0, 5.0, 15.0, 19.5, 20.1, 20.0, 19.9, 20.0, 19.8, 10.0, 1.0],  # Chamber Pressure
        'thrust': [0, 100, 400, 490, 505, 500, 495, 500, 490, 200, 0],  # Thrust
        'pt_ox': [30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]  # Oxidizer Tank Pressure
    }
    df = pd.read_csv(csv_path) if csv_path else pd.DataFrame(data)
    return df


# 2. PLOT GENERATION
def generate_plots(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Chamber Pressure & Thrustl
    fig, ax1 = plt.subplots(figsize=(8, 4))

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Chamber Pressure (bar)', color=color)
    ax1.plot(df['time'], df['pc'], color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Thrust (N)', color=color)
    ax2.plot(df['time'], df['thrust'], color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Static Fire: Pc & Thrust")
    plt.tight_layout()
    path_pc = os.path.join(output_dir, 'plot_pc.png')
    plt.savefig(path_pc, dpi=300)
    plt.close()

    # Plot 2: Tank Pressures
    plt.figure(figsize=(8, 4))
    plt.plot(df['time'], df['pt_ox'], label='Oxidizer Tank', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (bar)')
    plt.title("Feed System Pressures")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path_tank = os.path.join(output_dir, 'plot_tank.png')
    plt.savefig(path_tank, dpi=300)
    plt.close()

    # Return absolute paths for the HTML renderer
    return os.path.abspath(path_pc), os.path.abspath(path_tank)


# 3. REPORT GENERATION
def create_pdf_report(template_path, data_context, output_pdf):
    # Load Markdown Template
    with open(template_path, 'r') as f:
        template_text = f.read()

    # Render Jinja2 (Fill placeholders)
    template = jinja2.Template(template_text)
    rendered_markdown = template.render(data_context)

    # Convert Markdown to HTML
    html_content = markdown.markdown(rendered_markdown, extensions=['tables'])

    # Add simple CSS for professional formatting
    css = CSS(string='''
        body { font-family: Helvetica, Arial, sans-serif; font-size: 12px; line-height: 1.5; }
        h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
        h2 { color: #e67e22; margin-top: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #eee; }
    ''')

    # Write PDF
    HTML(string=html_content, base_url=os.getcwd()).write_pdf(output_pdf, stylesheets=[css])
    print(f"Report generated: {output_pdf}")


# MAIN EXECUTION
if __name__ == "__main__":
    # A. Setup
    test_id = "SF-001-A"
    df = load_and_process_data(None)  # Pass None to use mock data

    # B. Generate Assets
    pc_plot_path, tank_plot_path = generate_plots(df, "report_assets")

    # C. Prepare Variables for Template
    context = {
        'test_id': test_id,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'engineer_name': "J. Kerman",
        'target_pressure': 20,
        'user_notes': "Ignition was slightly delayed. Smooth shutdown.",
        # Calculated metrics
        'max_pc': round(df['pc'].max(), 2),
        'max_thrust': round(df['thrust'].max(), 2),
        'burn_time': round(df[df['pc'] > 5]['time'].count() * 0.1, 1),  # Simple duration logic
        'isp': 210,  # Placeholder for complex calc
        # Image paths
        'plot_pressure_path': pc_plot_path,
        'plot_tank_path': tank_plot_path
    }

    # D. Build Report
    create_pdf_report('test_template.md', context, f'Report_{test_id}.pdf')