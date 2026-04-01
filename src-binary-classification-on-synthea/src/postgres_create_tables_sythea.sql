-- Ensure schema exists
CREATE SCHEMA IF NOT EXISTS synthea;

-- Drop tables (in dependency order)
DROP TABLE IF EXISTS synthea.observations CASCADE;
DROP TABLE IF EXISTS synthea.procedures CASCADE;
DROP TABLE IF EXISTS synthea.immunizations CASCADE;
DROP TABLE IF EXISTS synthea.medications CASCADE;
DROP TABLE IF EXISTS synthea.conditions CASCADE;
DROP TABLE IF EXISTS synthea.encounters CASCADE;
DROP TABLE IF EXISTS synthea.careplans CASCADE;
DROP TABLE IF EXISTS synthea.allergies CASCADE;
DROP TABLE IF EXISTS synthea.organizations CASCADE;
DROP TABLE IF EXISTS synthea.providers CASCADE;
DROP TABLE IF EXISTS synthea.patients CASCADE;

-- ============================================
-- PATIENTS
-- ============================================
CREATE TABLE synthea.patients (
    id VARCHAR(64) PRIMARY KEY,
    birthdate DATE,
    deathdate DATE,
    ssn VARCHAR(20),
    drivers VARCHAR(50),
    passport VARCHAR(50),
    prefix VARCHAR(20),
    first VARCHAR(50),
    last VARCHAR(50),
    suffix VARCHAR(20),
    maiden VARCHAR(50),
    marital VARCHAR(20),
    race VARCHAR(50),
    ethnicity VARCHAR(50),
    gender VARCHAR(10),
    birthplace VARCHAR(100),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(50),
    county VARCHAR(100),
    zip VARCHAR(20),
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    healthcare_expenses DOUBLE PRECISION,
    healthcare_coverage DOUBLE PRECISION
);

-- ============================================
-- ORGANIZATIONS
-- ============================================
CREATE TABLE synthea.organizations (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(50),
    zip VARCHAR(20),
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    phone VARCHAR(50),
    revenue DOUBLE PRECISION,
    utilization DOUBLE PRECISION
);

-- ============================================
-- PROVIDERS
-- ============================================
CREATE TABLE synthea.providers (
    id VARCHAR(64) PRIMARY KEY,
    organization_id VARCHAR(64),
    name VARCHAR(255),
    gender VARCHAR(10),
    specialty VARCHAR(100),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(50),
    zip VARCHAR(20),
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    utilization DOUBLE PRECISION,
    FOREIGN KEY (organization_id) REFERENCES synthea.organizations(id) ON DELETE SET NULL
);

-- ============================================
-- ENCOUNTERS
-- ============================================
CREATE TABLE synthea.encounters (
    id VARCHAR(64) PRIMARY KEY,
    start DATE,
    stop DATE,
    patient VARCHAR(64),
    organization VARCHAR(64),
    provider VARCHAR(64),
    payer VARCHAR(100),
    encounterclass VARCHAR(50),
    code VARCHAR(50),
    description TEXT,
    base_encounter_cost DOUBLE PRECISION,
    total_claim_cost DOUBLE PRECISION,
    payer_coverage DOUBLE PRECISION,
    reasoncode VARCHAR(50),
    reasondescription TEXT,
    FOREIGN KEY (patient) REFERENCES synthea.patients(id) ON DELETE CASCADE,
    FOREIGN KEY (organization) REFERENCES synthea.organizations(id) ON DELETE SET NULL,
    FOREIGN KEY (provider) REFERENCES synthea.providers(id) ON DELETE SET NULL
);

-- ============================================
-- CONDITIONS
-- ============================================
CREATE TABLE synthea.conditions (
    row_id SERIAL PRIMARY KEY,
    start DATE,
    stop DATE,
    patient VARCHAR(64),
    encounter VARCHAR(64),
    code VARCHAR(50),
    description TEXT,
    FOREIGN KEY (patient) REFERENCES synthea.patients(id) ON DELETE CASCADE,
    FOREIGN KEY (encounter) REFERENCES synthea.encounters(id) ON DELETE CASCADE
);

-- ============================================
-- PROCEDURES
-- ============================================
CREATE TABLE synthea.procedures (
    row_id SERIAL PRIMARY KEY,
    date DATE,
    patient VARCHAR(64),
    encounter VARCHAR(64),
    code VARCHAR(50),
    description TEXT,
    base_cost DOUBLE PRECISION,
    reasoncode VARCHAR(50),
    reasondescription TEXT,
    FOREIGN KEY (patient) REFERENCES synthea.patients(id) ON DELETE CASCADE,
    FOREIGN KEY (encounter) REFERENCES synthea.encounters(id) ON DELETE CASCADE
);

-- ============================================
-- MEDICATIONS
-- ============================================
CREATE TABLE synthea.medications (
    row_id SERIAL PRIMARY KEY,
    start DATE,
    stop DATE,
    patient VARCHAR(64),
    payer VARCHAR(100),
    encounter VARCHAR(64),
    code VARCHAR(50),
    description TEXT,
    base_cost DOUBLE PRECISION,
    payer_coverage DOUBLE PRECISION,
    dispenses DOUBLE PRECISION,
    totalcost DOUBLE PRECISION,
    reasoncode VARCHAR(50),
    reasondescription TEXT,
    FOREIGN KEY (patient) REFERENCES synthea.patients(id) ON DELETE CASCADE,
    FOREIGN KEY (encounter) REFERENCES synthea.encounters(id) ON DELETE CASCADE
);

-- ============================================
-- IMMUNIZATIONS
-- ============================================
CREATE TABLE synthea.immunizations (
    row_id SERIAL PRIMARY KEY,
    date DATE,
    patient VARCHAR(64),
    encounter VARCHAR(64),
    code VARCHAR(50),
    description TEXT,
    base_cost DOUBLE PRECISION,
    FOREIGN KEY (patient) REFERENCES synthea.patients(id) ON DELETE CASCADE,
    FOREIGN KEY (encounter) REFERENCES synthea.encounters(id) ON DELETE CASCADE
);

-- ============================================
-- OBSERVATIONS
-- ============================================
CREATE TABLE synthea.observations (
    row_id SERIAL PRIMARY KEY,
    date DATE,
    patient VARCHAR(64),
    encounter VARCHAR(64),
    code VARCHAR(50),
    description TEXT,
    value VARCHAR(255),
    units VARCHAR(50),
    type VARCHAR(50),
    FOREIGN KEY (patient) REFERENCES synthea.patients(id) ON DELETE CASCADE,
    FOREIGN KEY (encounter) REFERENCES synthea.encounters(id) ON DELETE CASCADE
);

-- ============================================
-- CAREPLANS
-- ============================================
CREATE TABLE synthea.careplans (
    id VARCHAR(64) PRIMARY KEY,
    start DATE,
    stop DATE,
    patient VARCHAR(64),
    encounter VARCHAR(64),
    code VARCHAR(50),
    description TEXT,
    reasoncode VARCHAR(50),
    reasondescription TEXT,
    FOREIGN KEY (patient) REFERENCES synthea.patients(id) ON DELETE CASCADE,
    FOREIGN KEY (encounter) REFERENCES synthea.encounters(id) ON DELETE CASCADE
);

-- ============================================
-- ALLERGIES
-- ============================================
CREATE TABLE synthea.allergies (
    row_id SERIAL PRIMARY KEY,
    start DATE,
    stop DATE,
    patient VARCHAR(64),
    encounter VARCHAR(64),
    code VARCHAR(50),
    description TEXT,
    FOREIGN KEY (patient) REFERENCES synthea.patients(id) ON DELETE CASCADE,
    FOREIGN KEY (encounter) REFERENCES synthea.encounters(id) ON DELETE CASCADE
);
