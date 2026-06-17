# ============================================================
# terraform/main.tf
# ------------------------------------------------------------
# GCP infrastructure for CallOS: Cloud Run + Cloud SQL + Redis
#
# What it does:
#   Provisions the production stack — the Cloud Run API service, a
#   PostgreSQL (pgvector) instance, a Redis cache, and the DB-URL
#   secret the service reads at runtime.
#
# How it fits in CallOS:
#   `terraform -chdir=terraform apply` stands up everything the
#   README's Phase 5 deployment needs. Image is built by cloudbuild.
# ============================================================

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# DB connection string, stored in Secret Manager (never in env files).
resource "google_secret_manager_secret" "db_url" {
  secret_id = "callos-db-url"
  replication {
    auto {}
  }
}

# Serverless API — scales to zero, reads the DB secret at runtime.
resource "google_cloud_run_v2_service" "callos_api" {
  name     = "callos-api"
  location = var.region

  template {
    containers {
      image = "gcr.io/${var.project_id}/callos-api:latest"

      env {
        name = "DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_url.secret_id
            version = "latest"
          }
        }
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
  }
}

# Managed PostgreSQL 16 with pgvector for the KB embeddings.
resource "google_sql_database_instance" "callos_db" {
  name             = "callos-postgres"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier = var.db_tier
    database_flags {
      name  = "cloudsql.enable_pgvector"
      value = "on"
    }
  }
}

# Redis for live session state + agent pub/sub.
resource "google_redis_instance" "callos_cache" {
  name           = "callos-redis"
  tier           = "BASIC"
  memory_size_gb = 1
  region         = var.region
}
