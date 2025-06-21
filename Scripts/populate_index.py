import os
from dotenv import load_dotenv
import openai

# **Import the Pinecone class**, not the module
from pinecone import Pinecone

#load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"),
               environment=os.getenv("PINECONE_ENV"))

#get index
index_name = "resume-index"
if index_name not in pc.list_indexes().names():pc.create_index(name=index_name, dimension=384, metric="cosine")
index = pc.Index(index_name)

#Define resume chunks

sections = [
    {"id": "executive profile","text": "Strategic and hands-on technical leader whose data and analytics initiatives have directly generated $300M+ in lifetime revenue. Expert in building modern data stack infrastructure and scaling data capabilities in fast-growing technology environments. Proven track record transforming fragmented data silos into AI-driven self-service analytics platforms that drive executive decision-making across Sales, Marketing, Engineering, Finance, and Product teams."},
    {"id": "role_head_of_data",
     "text": """KLOUD9 LLC, Severna Park, MD (Remote)
HEAD OF DATA (DIRECTOR) | August 2023 – Present
Delivered $25M+ in measurable impact by transforming enterprise data environments into secure, cloud-native analytics platforms with real-time processing, governance, and AI-readiness. Scaled a remote 7-person team with 100% retention, while maintaining hands-on technical leadership and managing delivery budget.
•	Defined a multi-year enterprise data strategy integrating governance, AI readiness, and modern ELT pipelines, resulting in $35M in executive-approved funding to scale analytics capabilities.
•	Rearchitected ETL processes, improving data pipeline latency by 40% while reducing licensing costs $300K annually.
•	Led enterprise-wide Snowflake data warehouse implementation unifying data lakes while enabling data modeling and real-time data processing capabilities to enhance internal analytics capabilities for 500+ users.
•	Rolled out Alation metadata management and comprehensive data governance and data stewardship frameworks ensuring data quality, security, and compliance, cutting data-definition inquiries by 90%.

"""
     },
    {"id": "role_senior_manager",
     "text":"""SLALOM, Philadelphia, PA
DATA AND ANALYTICS SENIOR PRINCIPAL (SENIOR MANAGER) | January 2021 – August 2023
DATA AND ANALYTICS CONSULTANT | October 2017 – January 2021
Promoted from Consultant to Senior Principal, leading initiatives delivering $35M+ in new revenue and $46M in cost savings for Fortune 500 and fast-growing startup clients across retail, consumer packaged goods, healthcare, life sciences, and financial services. Built Philadelphia market Data Visualization practice, scaling to a 9-person team while working across practices to deliver enterprise analytics transformations. 
•	Led predictive analytics inventory optimization through Databricks/Snowflake/Tableau, achieving a 92% recall and reducing out-of-stock events and generating $25M in documented cost savings.
•	Built self-service data platforms enabling 300+ business users to access analytics independently, reducing insight delivery from 14 days to real-time.
•	Aligned 22 enterprise KPIs with C-suite OKRs and developed automated reporting roadmap, cutting quarterly operations review prep by 300 hours annually.
•	Established Data & Analytics Centers of Excellence with governance frameworks that provided training and data access to over 200 users while maintaining compliance and data quality.
•	Researched emerging data and technology trends to identify and develop new go-to-market capability offerings, expanding service differentiation and generating $7.5M in revenue for Philadelphia Market.

"""
     },
     {"id": "role_data_visaulization_lead",
      "text":"""WOLTERS KLUWER HEALTH, Philadelphia, PA
DATA VISUALIZATION LEAD | November 2014 – October 2017
Built enterprise reporting capabilities from the ground up for a $2B B2B healthcare SaaS company serving 1,200+ employees across multiple markets. Developed 40+ interactive dashboards that identified $20M in new revenue opportunities and improved marketing campaign ROI by 50%.
•	Created sales analytics environment that identified and helped capture $20M in previously hidden revenue opportunities across product portfolio.
•	Implemented comprehensive campaign reporting for print and digital media, increasing ROI 50%.
•	Pioneered embedded Tableau analytics within Salesforce CRM, enabling real-time decision-making for 200+ sales professionals.
•	Partnered with the engineering team to integrate disparate data sources into Amazon Redshift data warehouse.
•	Built and maintained 40+ interactive dashboards serving Product, Sales, Marketing, and HR teams, establishing self-service analytics culture.
•	Managed two-person analyst team, responsible for upskilling and handling business analytics requests.

"""
         
     },

     {"id": "education",
      "text": """•	Master of Science in Information Systems | Penn State University
•	Bachelor of Arts in Journalism | Penn State University"""
},
    {"id": "skills_and_technologies",
     "text": """Data Engineering & Architecture:
•	Enterprise data strategy, cloud-based data architectures
•	SQL, Python, DBT, Azure Data Factory, ETL/ELT pipelines
•	Data engineering, analytics engineering, data modeling
•	Enterprise data architecture, DataOps, CI/CD

Cloud Data Platforms:
•	Snowflake, Databricks, Amazon Redshift, SQL Server
•	AWS (Certified Cloud Practitioner), Azure
•	Modern data stack implementation and optimization

Analytics & Business Intelligence:
•	Tableau (Qualified Associate), Power BI, Looker, Sigma
•	Self-service analytics platforms, embedded analytics
•	Data visualization, reporting automation

Data Governance & Quality:
•	Data governance frameworks, data quality standards
•	Alation, metadata management, compliance
•	Data catalog implementation, unified data models
•	Master data management (MDM), data stewardship

"""}
]

# Embed & upsert all chunks

vectors = []
for section in sections:
    # 1) Call Embeddings.create with a capital “E”
    resp = openai.embeddings.create(
        input=section["text"],
        model="text-embedding-3-small"
    )

    # 2) Access the first embedding via attributes, not resp["data"]
    emb = resp.data[0].embedding

    vectors.append((section["id"], emb, {"text": section["text"]}))

    index.upsert(vectors)
    print(f"Upserted{len(vectors)} vectors into '{index_name}'.")
