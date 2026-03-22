-- benchmark_results table for GPUSCALE
-- Run this in the Supabase SQL Editor to create the table.

-- Enable uuid-ossp if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS benchmark_results (
    -- Identity
    id                          uuid            PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at                  timestamptz     NOT NULL DEFAULT now(),

    -- GPU
    gpu_name                    text            NOT NULL,
    gpu_vram_gb                 numeric         NOT NULL CHECK (gpu_vram_gb > 0),
    gpu_count                   integer         NOT NULL CHECK (gpu_count >= 1 AND gpu_count <= 64),

    -- Environment
    provider                    text            NOT NULL CHECK (provider IN ('local', 'vast.ai', 'runpod')),
    engine                      text            NOT NULL CHECK (engine IN ('llama.cpp', 'vllm')),

    -- Model
    model_name                  text            NOT NULL,
    quantization                text            NOT NULL,

    -- Workload
    workload_version            text            NOT NULL,
    workload_config             jsonb,

    -- Performance metrics
    tokens_per_sec              numeric         NOT NULL CHECK (tokens_per_sec > 0),
    time_to_first_token_ms      numeric         NOT NULL CHECK (time_to_first_token_ms >= 0),
    prompt_eval_tokens_per_sec  numeric         CHECK (prompt_eval_tokens_per_sec >= 0),

    -- Resource metrics
    peak_vram_mb                numeric         CHECK (peak_vram_mb >= 0),
    avg_power_draw_w            numeric         CHECK (avg_power_draw_w >= 0),
    peak_power_draw_w           numeric         CHECK (peak_power_draw_w >= 0),
    avg_gpu_util_pct            numeric         CHECK (avg_gpu_util_pct >= 0 AND avg_gpu_util_pct <= 100),
    avg_gpu_temp_c              numeric         CHECK (avg_gpu_temp_c >= 0 AND avg_gpu_temp_c <= 150),
    total_wall_time_s           numeric         CHECK (total_wall_time_s > 0),

    -- Versions
    engine_version              text,

    -- Host (local only)
    host_os                     text,
    host_kernel                 text,
    host_driver_version         text,

    -- Container
    container_image             text,
    container_driver_version    text,

    -- Raw output
    raw_output                  jsonb,

    -- Moderation
    flagged                     boolean         NOT NULL DEFAULT false,

    -- Cross-field constraint
    CONSTRAINT peak_gte_avg_power CHECK (
        peak_power_draw_w IS NULL
        OR avg_power_draw_w IS NULL
        OR peak_power_draw_w >= avg_power_draw_w
    )
);

-- Indexes on commonly filtered columns
CREATE INDEX IF NOT EXISTS idx_br_gpu_name      ON benchmark_results (gpu_name);
CREATE INDEX IF NOT EXISTS idx_br_model_name    ON benchmark_results (model_name);
CREATE INDEX IF NOT EXISTS idx_br_engine        ON benchmark_results (engine);
CREATE INDEX IF NOT EXISTS idx_br_provider      ON benchmark_results (provider);
CREATE INDEX IF NOT EXISTS idx_br_quantization  ON benchmark_results (quantization);
CREATE INDEX IF NOT EXISTS idx_br_created_at    ON benchmark_results (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_br_flagged       ON benchmark_results (flagged) WHERE flagged = true;

-- Enable Row Level Security (public read, service-role write)
ALTER TABLE benchmark_results ENABLE ROW LEVEL SECURITY;

-- Anyone can read
CREATE POLICY "Public read access"
    ON benchmark_results
    FOR SELECT
    USING (true);

-- Only service role can insert
CREATE POLICY "Service role insert"
    ON benchmark_results
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

-- Only service role can update (for flagging)
CREATE POLICY "Service role update"
    ON benchmark_results
    FOR UPDATE
    USING (auth.role() = 'service_role');

-- Only service role can delete
CREATE POLICY "Service role delete"
    ON benchmark_results
    FOR DELETE
    USING (auth.role() = 'service_role');
