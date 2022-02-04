--Create tablespace for covid19_data_analysis database	
CREATE TABLESPACE covid19_data_analysis_space LOCATION 'C:\Postgres\data';

--Set the default tablespace
SET default_tablespace = covid19_data_analysis_space;

--Create schema for covid19_data_analysis_space database
CREATE SCHEMA covid
    AUTHORIZATION postgres;

COMMENT ON SCHEMA covid
    IS 'schema created for covid data analysis as part of Module 20 group project';

GRANT ALL ON SCHEMA public TO PUBLIC;

GRANT ALL ON SCHEMA public TO postgres;

-- Set default schema to covid 
ALTER DATABASE covid19_data_analysis SET search_path TO covid;

--Set default schema for use
SET search_path TO covid;

CREATE TABLE IF NOT EXISTS covid.covid_dataset
(
    id varchar,
    sex int,
    patient_type int,
    entry_date varchar,
    date_symptoms varchar,
    date_died varchar,
    intubed int,
    pneumonia int,
    age int,
    pregnancy int,
    diabetes int,
    copd int,
    asthma int,
    inmsupr int,
    hypertension int,
    other_disease int,
    cardiovascular int,
    obesity int,
    renal_chronic int,
    tobacco int,
    contact_other_covid int,
    covid_res int,
    icu int
);

CREATE TABLE IF NOT EXISTS covid.catalogs
(
    column_name varchar(30),
	column_value int,
	column_value_description varchar(50),
	CONSTRAINT fk_column_name 
		FOREIGN KEY(column_name) 
	  		REFERENCES column_description(column_name)
);

CREATE TABLE IF NOT EXISTS covid.column_description
(
    index smallint,
	column_name varchar(30) PRIMARY KEY,
	column_description varchar(200)
);


CREATE TABLE IF NOT EXISTS covid.clean_covid_dataset
(
    id varchar PRIMARY KEY,
    sex int,
    patient_type int,
    entry_date varchar,
    date_symptoms varchar,
    date_died varchar,
    intubed int,
    pneumonia int,
    age int,
    pregnancy int,
    diabetes int,
    copd int,
    asthma int,
    inmsupr int,
    hypertension int,
    other_disease int,
    cardiovascular int,
    obesity int,
    renal_chronic int,
    tobacco int,
    contact_other_covid int,
    covid_res int,
    icu int,
    survived int
);

select * from covid_dataset;

select * from column_description;

select * from catalogs;

select * from covid.clean_covid_dataset;

select d.id,d.sex,c.column_value_description from clean_covid_dataset d inner join catalogs c on d.sex = cast(c.column_value as bigint)
where c.column_name = 'sex';


