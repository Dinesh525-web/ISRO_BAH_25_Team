# Staging Environment Configuration
project_name = "mosdac-ai"
environment  = "staging"
aws_region   = "us-east-1"
domain_name  = "staging.mosdac-ai.com"

# VPC Configuration
vpc_cidr         = "10.1.0.0/16"
private_subnets  = ["10.1.1.0/24", "10.1.2.0/24", "10.1.3.0/24"]
public_subnets   = ["10.1.101.0/24", "10.1.102.0/24", "10.1.103.0/24"]
intra_subnets    = ["10.1.51.0/24", "10.1.52.0/24", "10.1.53.0/24"]
single_nat_gateway = true

# EKS Configuration
kubernetes_version = "1.28"
node_instance_types = ["m5.large"]

node_group_min_size     = 2
node_group_max_size     = 10
node_group_desired_size = 3

spot_instance_types = ["m5.large", "m5a.large", "m4.large"]
spot_min_size       = 1
spot_max_size       = 5
spot_desired_size   = 2

# Database Configuration
postgres_version = "15.4"
db_instance_class = "db.r6g.large"
db_allocated_storage = 100
db_max_allocated_storage = 500
db_backup_retention_period = 7

# Redis Configuration
redis_node_type = "cache.r6g.large"
redis_num_cache_clusters = 2

# Additional Tags
additional_tags = {
  CostCenter   = "Engineering"
  Owner        = "GravitasOps"
  Environment  = "Staging"
}

