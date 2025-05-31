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
    {"id": "executive profile","text": "Strategic data leader with over a decade of experience establishing cloud-native data infrastructures, centralizing disparate data sources, and delivering actionable insights across organizations. Proven expertise in designing end-to-end data architectures, implementing self-service analytics platforms, and partnering with cross-functional stakeholders to drive data-informed decision making. Adept at building scalable analytics functions that align with business objectives and support rapid growth"},
    {"id": "role_head_of_data",
     "text": """KLOUD9 LLC, Severna Park, MD (Remote)
HEAD OF DATA | August 2023 – Present
Strategic leadership role driving data transformation across U.S. clients, fostering cross-functional partnerships with C-level executives to align business priorities with comprehensive data strategies and implement data engineering and analytics capabilities that drive significant business improvement.
•	Conducted gap analysis and maturity assessment across the enterprise and developed 3-year strategic data and analytics roadmap to modernize data technology architecture and capabilities, influencing $35 million budget.
•	Designed enhanced reference architecture for end-to-end data ingestion and reporting, utilizing Python, Azure Data Factory, Snowflake, Alation, Tableau and Power BI, improving platform performance by 75% and reducing redundant costs by $300K annually.
•	Led hands-on development of Snowflake enterprise data warehouse to enable centralized source of truth and data analysis across domains, generating $15 million in revenue and efficiency gains.
•	Orchestrated cross-functional data governance framework spanning 15 departments, establishing enterprise-wide data compliance, integrity standards, and data catalog capabilities.
•	Partnered with data science team to conduct workshops with business to identify use-cases and support development proof of concept scoring model.
•	Managed a distributed team of 7 data engineers and analysts, overseeing workloads and ensuring consistent delivery of high-quality data solutions.
"""
     },
    {"id": "role_senior_manager",
     "text":"""SLALOM, Philadelphia, PA
DATA AND ANALYTICS SENIOR MANAGER | January 2021 – August 2023
DATA AND ANALYTICS CONSULTANT | October 2017 – January 2021
Responsible for managing multiple projects and enterprise-scale data solutions across Fortune 500 clients, driving analytics innovation, and leading high-performance consulting teams.
•	Owned design and delivery of modern data and analytics solutions for Fortune 500 clients in the Philadelphia market across healthcare, life sciences, retail, consumer goods, and financial services.
•	Collaborated with stakeholders to discern business-critical Key Performance Indicators and create a long-term measurement strategy aligned with organizational objectives.
•	Led implementation of self-service analytics platforms, building data warehouse and reporting environments and establishing an operating model to for generating self-service insights for over 500 users.
•	Led proof-of-concept to develop predictive and prescriptive inventory models through Databricks and visualize results via Tableau, reducing out-of-stock events and driving $25M in cost savings.
•	Conducted internal and external training and enablement on data and analytics best practices to improve data fluency for non-technical stakeholders.
•	Oversaw creation of a Data and Analytics Center of Excellence, providing processes and operating framework for organization to democratize data in a governed fashion and infuse data-driven decision-making into day-to-day operations.
•	Managed operational metrics around budget, sales, managed revenue, and team performance through enterprise technologies including Salesforce and Workday. 
•	Built the Data Visualization practice, interviewing candidates and managing a team of 9 consultants and managers, providing mentorship and supporting career growth.
"""
     },
     {"id": "role_data_visaulization_lead",
      "text":"""WOLTERS KLUWER HEALTH, Philadelphia, PA
DATA VISUALIZATION LEAD | November 2014 – October 2017
Enterprise-level data visualization role responsible for creating strategic analytics solutions across multiple business functions, driving insights and operational efficiency.
•	Led development and maintenance of 40+ interactive dashboards across Product, Sales, Marketing, and HR teams.
•	Created a sales analytics environment, providing data analysis and exploration to identify $13.5 million in new revenue opportunities.
•	Implemented a marketing analytics solution for print and digital media campaigns, increasing campaign ROI by 50%.
•	Pioneered embedded Tableau analytics integration within Salesforce CRM, improving the user experience and enabling real-time decision-making.
"""
         
     },

     {"id": "education",
      "text": """•	Master of Science in Information Systems | Penn State University
•	Bachelor of Arts in Journalism | Penn State University"""
},
    {"id": "skills_and_technologies",
     "text": """Data Strategy and Transformation
•	Data Infrastructure Design and Implementation
•	Data Warehouse Architecture and Management
•	Enterprise Data Governance
•	Self-Service Analytics and Dashboard Development
•	Data Strategy Alignment
Technology
•	Cloud Platforms: AWS (Certified Cloud Practitioner), Azure
•	Data Engineering: SQL, DBT, Python, Azure Data Factory
•	Cloud Warehousing: Snowflake, Redshift, Databricks, SQL Server
•	BI Tools: Tableau (Qualified Associate), Power BI, Sigma, Looker
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
