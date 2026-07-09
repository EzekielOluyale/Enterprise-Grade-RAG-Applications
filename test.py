import logfire
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.cleaner import clean_amusEcode_data 

logfire.configure(service_name="enterprise-ingestion-service")

raw = parse_pdf("DATA/Ezekiel_Oluyale/amusEcode_Corporate_Profile.pdf")
clean = clean_amusEcode_data(raw)

# Save both so you can visually compare them
with open("DATA/noisy_data/noise.txt", "w", encoding="utf-8") as f:
    f.write(raw)

with open("DATA/true_data/true.txt", "w", encoding="utf-8") as f:
    f.write(clean)