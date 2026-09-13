CREATE TABLE IF NOT EXISTS infinite_sets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    series_type VARCHAR(50) NOT NULL CHECK (series_type IN ('harmonic','alternating','geometric','custom')),
    parameters JSONB NOT NULL DEFAULT '{}', description TEXT,
    convergence_info JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS set_elements (
    id BIGSERIAL PRIMARY KEY,
    set_id UUID NOT NULL REFERENCES infinite_sets(id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0), value DOUBLE PRECISION NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(set_id, position)
);
CREATE TABLE IF NOT EXISTS compensation_pairs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    set1_id UUID NOT NULL REFERENCES infinite_sets(id) ON DELETE CASCADE,
    set2_id UUID NOT NULL REFERENCES infinite_sets(id) ON DELETE CASCADE,
    compensation_quality DOUBLE PRECISION NOT NULL CHECK (compensation_quality BETWEEN 0 AND 1),
    method_used VARCHAR(50) NOT NULL, tolerance DOUBLE PRECISION NOT NULL CHECK(tolerance > 0),
    metadata JSONB NOT NULL DEFAULT '{}', created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(set1_id, set2_id)
);
CREATE TABLE IF NOT EXISTS operation_logs (
    id BIGSERIAL PRIMARY KEY, operation_id UUID NOT NULL,
    operation_type VARCHAR(50) NOT NULL, parameters JSONB NOT NULL DEFAULT '{}', result JSONB,
    status VARCHAR(20) NOT NULL CHECK(status IN ('success','failed','partial','timeout')),
    execution_time DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK(execution_time >= 0),
    error_message TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_infinite_sets_created_at ON infinite_sets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_compensation_pairs_set1_id ON compensation_pairs(set1_id);
CREATE INDEX IF NOT EXISTS idx_compensation_pairs_set2_id ON compensation_pairs(set2_id);
CREATE INDEX IF NOT EXISTS idx_operation_logs_operation_id ON operation_logs(operation_id);
