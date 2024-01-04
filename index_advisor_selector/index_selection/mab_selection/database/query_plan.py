import xml.etree.ElementTree as ET

ns = {"sp": "http://schemas.microsoft.com/sqlserver/2004/07/showplan"}
physical_operations = {"Index Seek", "Index Scan", "Clustered Index Scan", "Clustered Index Seek"}


class QueryPlan:
    def __init__(self, xml_string):
        self.estimated_rows = 0
        self.est_statement_sub_tree_cost = 0
        self.elapsed_time = 0
        self.cpu_time = 0
        self.non_clustered_index_usage = list()
        self.clustered_index_usage = list()

        root = ET.fromstring(xml_string)
        stmt_simple = root.find(".//sp:StmtSimple", ns)
        self.estimated_rows = stmt_simple.attrib.get("StatementEstRows")
        self.est_statement_sub_tree_cost = stmt_simple.attrib.get("StatementSubTreeCost")

        query_stats = root.find(".//sp:QueryTimeStats", ns)
        if query_stats is not None:
            self.cpu_time = query_stats.attrib.get("CpuTime")
            self.elapsed_time = float(query_stats.attrib.get("ElapsedTime")) / 1000

        rel_ops = root.findall(".//sp:RelOp", ns)
        total_po_sub_tree_cost = 0
        total_po_actual = 0
        # Get the sum of sub tree cost for physical operations
        # (assumption: sub tree cost is dominated by the physical operations)
        for rel_op in rel_ops:
            temp_act_elapsed_time = 0
            if rel_op.attrib.get("PhysicalOp") in physical_operations:
                total_po_sub_tree_cost += float(rel_op.attrib.get("EstimatedTotalSubtreeCost"))
                runtime_thread_information = rel_op.findall(".//sp:RunTimeCountersPerThread", ns)
                for thread_info in runtime_thread_information:
                    temp_act_elapsed_time = max(
                        int(thread_info.attrib.get("ActualElapsedms")) if thread_info.attrib.get(
                            "ActualRowsRead") is not None else 0, temp_act_elapsed_time)
                total_po_actual += temp_act_elapsed_time / 1000

        # Now for each rel operator we estimate the elapsed time using the sub tree costs
        for rel_op in rel_ops:
            rows_read = 0
            act_rel_op_elapsed_time = 0
            if rel_op.attrib.get("PhysicalOp") in physical_operations:
                runtime_thread_information = rel_op.findall(".//sp:RunTimeCountersPerThread", ns)
                for thread_info in runtime_thread_information:
                    rows_read += int(thread_info.attrib.get("ActualRowsRead")) if thread_info.attrib.get(
                        "ActualRowsRead") is not None else 0
                    act_rel_op_elapsed_time = max(
                        int(thread_info.attrib.get("ActualElapsedms")) if thread_info.attrib.get(
                            "ActualElapsedms") is not None else 0, act_rel_op_elapsed_time)
            act_rel_op_elapsed_time = act_rel_op_elapsed_time / 1000
            # act_rel_op_elapsed_time = float(self.elapsed_time) * (act_rel_op_elapsed_time/total_po_actual) if total_po_actual > 0 else 0
            # We can either use act_rel_op_elapsed_time or po_elapsed_time for the elapsed time
            if rows_read == 0:
                rows_read = float(rel_op.attrib.get("EstimatedRowsRead")) if rel_op.attrib.get(
                    "EstimatedRowsRead") else 0
            rows_output = float(rel_op.attrib.get("EstimateRows"))
            if rel_op.attrib.get("PhysicalOp") in physical_operations:
                po_subtree_cost = float(rel_op.attrib.get("EstimatedTotalSubtreeCost"))
                po_elapsed_time = float(self.elapsed_time) * (po_subtree_cost / total_po_sub_tree_cost)
                po_cpu_time = float(self.cpu_time) * (
                        po_subtree_cost / float(self.est_statement_sub_tree_cost))
                po_index_scan = rel_op.find(".//sp:IndexScan", ns)
                if rel_op.attrib.get("PhysicalOp") in {"Index Seek", "Index Scan"}:
                    po_index = po_index_scan.find(".//sp:Object", ns).attrib.get("Index").strip("[]")
                    self.non_clustered_index_usage.append(
                        (po_index, act_rel_op_elapsed_time, po_cpu_time, po_subtree_cost, rows_read, rows_output))
                elif rel_op.attrib.get("PhysicalOp") in {"Clustered Index Scan", "Clustered Index Seek"}:
                    table = po_index_scan.find(".//sp:Object", ns).attrib.get("Table").strip("[]")
                    self.clustered_index_usage.append(
                        (table, act_rel_op_elapsed_time, po_cpu_time, po_subtree_cost, rows_read, rows_output))


class QueryPlanPG:
    def __init__(self, plan_json):
        self.estimated_rows = 0
        self.est_statement_sub_tree_cost = 0
        self.elapsed_time = 0
        self.cpu_time = 0
        self.non_clustered_index_usage = list()
        self.clustered_index_usage = list()

        self.estimated_rows = plan_json["Plan"]["Plan Rows"]
        self.est_statement_sub_tree_cost = plan_json["Plan"]["Total Cost"]

        if "Actual Total Time" in plan_json.keys():
            self.cpu_time = plan_json["Plan"]["Actual Total Time"]
            self.elapsed_time = float(plan_json["Plan"]["Actual Total Time"]) / 1000

        rel_ops = [plan_json["Plan"]]
        ops_stack = [plan_json["Plan"]]
        while len(ops_stack) != 0:
            ops = ops_stack.pop(-1)
            if "Plans" in ops.keys():
                for op in ops["Plans"]:
                    rel_ops.append(op)
                    ops_stack.append(op)

        total_po_sub_tree_cost, total_po_actual = 0, 0
        # Get the sum of sub tree cost for physical operations
        # (assumption: sub tree cost is dominated by the physical operations)
        for rel_op in rel_ops:
            temp_act_elapsed_time = 0
            if "Index" in rel_op["Node Type"]:
                total_po_sub_tree_cost += float(rel_op["Total Cost"])
                # runtime_thread_information = rel_op.findall(".//sp:RunTimeCountersPerThread", ns)
                # for thread_info in runtime_thread_information:
                #     temp_act_elapsed_time = max(
                #         int(thread_info.attrib.get("ActualElapsedms")) if thread_info.attrib.get(
                #             "ActualRowsRead") is not None else 0, temp_act_elapsed_time)
                total_po_actual += temp_act_elapsed_time / 1000

        # Now for each rel operator we estimate the elapsed time using the sub tree costs
        for rel_op in rel_ops:
            rows_read = 0
            act_rel_op_elapsed_time = 0
            if "Index" in rel_op["Node Type"]:
                total_po_sub_tree_cost += float(rel_op["Plan Rows"])

            # if rel_op.attrib.get("PhysicalOp") in physical_operations:
            #     runtime_thread_information = rel_op.findall(".//sp:RunTimeCountersPerThread", ns)
            #     for thread_info in runtime_thread_information:
            #         rows_read += int(thread_info.attrib.get("ActualRowsRead")) if thread_info.attrib.get(
            #             "ActualRowsRead") is not None else 0
            #         act_rel_op_elapsed_time = max(
            #             int(thread_info.attrib.get("ActualElapsedms")) if thread_info.attrib.get(
            #                 "ActualElapsedms") is not None else 0, act_rel_op_elapsed_time)

            act_rel_op_elapsed_time = act_rel_op_elapsed_time / 1000
            # act_rel_op_elapsed_time = float(self.elapsed_time) * (act_rel_op_elapsed_time/total_po_actual) if total_po_actual > 0 else 0

            # We can either use act_rel_op_elapsed_time or po_elapsed_time for the elapsed time
            if rows_read == 0:
                rows_read = float(rel_op["Plan Rows"])
            rows_output = float(rel_op["Plan Rows"])
            if "Index" in rel_op["Node Type"]:
                po_subtree_cost = float(rel_op["Total Cost"])
                po_elapsed_time = float(self.elapsed_time) * (po_subtree_cost / total_po_sub_tree_cost)
                po_cpu_time = float(self.cpu_time) * (
                        po_subtree_cost / float(self.est_statement_sub_tree_cost))

                if "_pkey" not in str(rel_op):
                    po_index = rel_op["Index Name"]
                    self.non_clustered_index_usage.append(
                        (po_index, act_rel_op_elapsed_time, po_cpu_time, po_subtree_cost, rows_read, rows_output))
                elif "_pkey" in str(rel_op):
                    if "Relation Name" not in rel_op.keys():
                        table = rel_op["Index Name"].split("_pkey")[0]
                    else:
                        table = rel_op["Relation Name"]

                    self.clustered_index_usage.append(
                        (table, act_rel_op_elapsed_time, po_cpu_time, po_subtree_cost, rows_read, rows_output))
