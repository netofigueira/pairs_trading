-- The dashboard is intentionally read-only.  Roles are environment-owned, so
-- grant only when the dedicated production reader exists.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'quant_dashboard_reader') THEN
        GRANT SELECT ON research.paper_run TO quant_dashboard_reader;
        GRANT SELECT ON research.paper_decision TO quant_dashboard_reader;
        GRANT SELECT ON research.paper_position TO quant_dashboard_reader;
        GRANT SELECT ON research.paper_mark TO quant_dashboard_reader;
    END IF;
END $$;
