# Enterprise extensions

The retrieval contract can wrap Azure AI Search, OpenSearch/Elasticsearch, pgvector, Pinecone, Qdrant or Weaviate. Keep one configuration working end to end before adding more adapters. Reuse the same labelled dataset to prove parity and quantify trade-offs.

At scale, partition or filter by `tenant_id`, `project_id`, `access_group`, `region`, document status and effective date. Authorisation claims must be applied inside the search query. Add negative tests proving that one tenant, project or group cannot retrieve another's chunks.

Millions of documents also require incremental indexing, content hashes, tombstones, schema/version migration, search capacity tests, cache policy, disaster recovery and label sampling from real traffic. Evaluation datasets need controlled refresh and reviewer agreement rather than automatic synthetic labels alone.

