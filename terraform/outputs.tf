# ============================================================
# terraform/outputs.tf
# ------------------------------------------------------------
# Outputs surfaced after `terraform apply` — the handful of values
# you need to point Twilio and clients at the deployed stack.
# ============================================================

output "api_url" {
  description = "Public URL of the Cloud Run API service."
  value       = google_cloud_run_v2_service.callos_api.uri
}

output "db_connection_name" {
  description = "Cloud SQL instance connection name (for the proxy)."
  value       = google_sql_database_instance.callos_db.connection_name
}

output "redis_host" {
  description = "Internal host of the Redis instance."
  value       = google_redis_instance.callos_cache.host
}

output "db_url_secret" {
  description = "Secret Manager secret id holding the DATABASE_URL."
  value       = google_secret_manager_secret.db_url.secret_id
}
