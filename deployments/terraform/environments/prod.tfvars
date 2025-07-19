# Production Environment Configuration
project_name = "mosdac-ai"
environment  = "production"
aws_region   = "us-east-1"
domain_name  = "mosdac-ai.com"

# VPC Configuration
vpc_cidr         = "10.0.0.0/16"
private_subnets  = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
public_subnets   = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
intra_subnets    = ["10.0.51.0/24", "10.0.52.0/24", "10.0.53.0/24"]
single_nat_gateway = false

# EKS Configuration
kubernetes_version = "1.28"
node_instance_types = ["m5.xlarge", "m5.2xlarge"]

node_group_min_size     = 3
node_group_max_size     = 20
node_group_desired_size = 5

spot_instance_types = ["m5.large", "m5a.large", "m4.large", "c5.large", "c5a.large"]
spot_min_size       = 2
spot_max_size       = 10
spot_desired_size   = 3

# Database Configuration
postgres_version = "15.4"
db_instance_class = "db.r6g.2xlarge"
db_allocated_storage = 200
db_max_allocated_storage = 2000
db_backup_retention_period = 30

# Redis Configuration
redis_node_type = "cache.r6g.2xlarge"
redis_num_cache_clusters = 3

# Additional Tags
additional_tags = {
  CostCenter   = "Engineering"
  Owner        = "GravitasOps"
  Backup       = "Required"
  Compliance   = "Required"
  Monitoring   = "Enhanced"
}
