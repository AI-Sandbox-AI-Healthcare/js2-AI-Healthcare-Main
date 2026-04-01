-- Ensure schema is set
SET search_path TO synthea;

-- ============================================
-- Disable foreign key constraints before import
-- ============================================
ALTER TABLE synthea.observations DISABLE TRIGGER ALL;
ALTER TABLE synthea.procedures DISABLE TRIGGER ALL;
ALTER TABLE synthea.immunizations DISABLE TRIGGER ALL;
ALTER TABLE synthea.medications DISABLE TRIGGER ALL;
ALTER TABLE synthea.conditions DISABLE TRIGGER ALL;
ALTER TABLE synthea.encounters DISABLE TRIGGER ALL;
ALTER TABLE synthea.careplans DISABLE TRIGGER ALL;
ALTER TABLE synthea.allergies DISABLE TRIGGER ALL;
ALTER TABLE synthea.providers DISABLE TRIGGER ALL;
ALTER TABLE synthea.organizations DISABLE TRIGGER ALL;
ALTER TABLE synthea.patients DISABLE TRIGGER ALL;

-- ============================================
-- Import tables in dependency order
-- ============================================

\COPY synthea.patients FROM '~/Desktop/10k_synthea_covid19_csv/patients.csv' WITH CSV HEADER QUOTE '"' DELIMITER ',';

\COPY synthea.organizations FROM '~/Desktop/10k_synthea_covid19_csv/organizations.csv' WITH CSV HEADER QUOTE '"' DELIMITER ',';
\COPY synthea.providers FROM '~/Desktop/10k_synthea_covid19_csv/providers.csv' WITH CSV HEADER QUOTE '"' DELIMITER ',';

\COPY synthea.encounters FROM '~/Desktop/10k_synthea_covid19_csv/encounters.csv' WITH CSV HEADER QUOTE '"' DELIMITER ',';
\COPY synthea.conditions (start, stop, patient, encounter, code, description) FROM '~/Desktop/10k_synthea_covid19_csv/conditions.csv' WITH CSV HEADER QUOTE '"' DELIMITER ',';
\COPY synthea.procedures (date, patient, encounter, code, description, base_cost, reasoncode, reasondescription) FROM '~/Desktop/10k_synthea_covid19_csv/procedures.csv' WITH CSV HEADER QUOTE '"' DELIMITER ',';
\COPY synthea.medications (start, stop, patient, payer, encounter, code, description, base_cost, payer_coverage, dispenses, totalcost, reasoncode, reasondescription) FROM '~/Desktop/10k_synthea_covid19_csv/medications.csv' WITH CSV HEADER QUOTE '"' DELIMITER ',';
\COPY synthea.immunizations (date, patient, encounter, code, description, base_cost) FROM '~/Desktop/10k_synthea_covid19_csv/immunizations.csv' WITH CSV HEADER QUOTE '"' DELIMITER ',';
\COPY synthea.observations (date, patient, encounter, code, description, value, units, type) FROM '~/Desktop/10k_synthea_covid19_csv/observations.csv' WITH CSV HEADER QUOTE '"' DELIMITER ',';
\COPY synthea.careplans FROM '~/Desktop/10k_synthea_covid19_csv/careplans.csv' WITH CSV HEADER QUOTE '"' DELIMITER ',';
\COPY synthea.allergies (start, stop, patient, encounter, code, description) FROM '~/Desktop/10k_synthea_covid19_csv/allergies.csv' WITH CSV HEADER QUOTE '"' DELIMITER ',';

-- ============================================
-- Re-enable foreign key constraints
-- ============================================
ALTER TABLE synthea.observations ENABLE TRIGGER ALL;
ALTER TABLE synthea.procedures ENABLE TRIGGER ALL;
ALTER TABLE synthea.immunizations ENABLE TRIGGER ALL;
ALTER TABLE synthea.medications ENABLE TRIGGER ALL;
ALTER TABLE synthea.conditions ENABLE TRIGGER ALL;
ALTER TABLE synthea.encounters ENABLE TRIGGER ALL;
ALTER TABLE synthea.careplans ENABLE TRIGGER ALL;
ALTER TABLE synthea.allergies ENABLE TRIGGER ALL;
ALTER TABLE synthea.providers ENABLE TRIGGER ALL;
ALTER TABLE synthea.organizations ENABLE TRIGGER ALL;
ALTER TABLE synthea.patients ENABLE TRIGGER ALL;
