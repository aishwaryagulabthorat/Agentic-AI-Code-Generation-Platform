# test_pipeline.py
from pipeline.core import instruction_parser_agent

if __name__ == "__main__":
    query = "Make a dashboard to list AWD inbound shipments with filters by status and date; drill into a shipment to see items."
    result = instruction_parser_agent(query)
    print("\n=== PARSER OUTPUT ===")
    print(result)
