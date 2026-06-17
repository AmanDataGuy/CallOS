# ============================================================
# terraform/variables.tf
# ------------------------------------------------------------
# Input variables for the CallOS GCP stack.
#
# Set these via terraform.tfvars or -var flags. project_id has no
# default on purpose so an apply can't target the wrong project.
# ============================================================

variable "project_id" {
  description = "GCP project id to deploy CallOS into."
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run, Cloud SQL, and Redis."
  type        = string
  default     = "us-central1"
}

variable "db_tier" {
  description = "Cloud SQL machine tier (db-g1-small is cheapest usable)."
  type        = string
  default     = "db-g1-small"
}
