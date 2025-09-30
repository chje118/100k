import pandas as pd
import copy

class SNOMEDCodes: 
    """ Class to handle SNOMED codes. """
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.first_letters = self._first_letters()

    def _first_letters(self) -> set:
        """ Get unique first letters of SNOMED codes. """
        return {code[0] for code in self.dataframe['SKSkode'].dropna().unique() if code}
    
    def get_first_letters(self) -> None:
        print('SNOMED letters:', self.first_letters)

    def get_letter_counts(self) -> None:
        prefixes = self.dataframe['SKSkode'].str[0].value_counts()
        print("Unique code prefixes in SKSkode: \n", prefixes)

    def get_codes_by_letter(self, letter: str) -> pd.DataFrame:
        """ Get SNOMED codes starting with a specific letter. """
        return self.dataframe[self.dataframe['SKSkode'].str.startswith(letter)].copy()

class SNOMEDHierarchy:
    """ Class to manage hierarchical structure of SNOMED codes. """
    def __init__(self, codes: pd.DataFrame, main_len: int = 3, sub_len: int = 4):
        assert main_len < sub_len, "main_len must be less than sub_len"
        self.codes = codes
        self.main_len = main_len
        self.sub_len = sub_len
        self.hierarchy = self._build_hierarchy()
        self.edited_hierarchy = copy.deepcopy(self.hierarchy)
        self._rebuild_code_to_region()

    def _build_hierarchy(self):
        """ Build hierarchical structure from codes DataFrame. """
        hierarchy = {}
        for _, row in self.codes.iterrows():
            code = row['SKSkode']
            text = row['Kodetekst']
            if not isinstance(code, str) or len(code) < self.main_len:
                print(f"Skipping invalid code: {code}")
                continue
            main = code[:self.main_len]
            sub = code[:self.sub_len]
            if main not in hierarchy:
                hierarchy[main] = {"name": text, "subregions": {}}
            if sub not in hierarchy[main]["subregions"]:
                hierarchy[main]["subregions"][sub] = {"name": text, "codes": []}
            hierarchy[main]["subregions"][sub]["codes"].append({"code": code, "text": text})
        return hierarchy

    def _rebuild_code_to_region(self):
        hierarchy = self.edited_hierarchy
        self.code_to_region = {}
        self.code_to_subregion = {}
        for main_region, region_dict in hierarchy.items():
            for subregion, sub_dict in region_dict["subregions"].items():
                for code_info in sub_dict["codes"]:
                    code = code_info["code"]
                    self.code_to_region[code] = main_region
                    self.code_to_subregion[code] = subregion

    def code_to_main_region_name(self, code):
        main_region = self.code_to_region.get(code)
        if main_region is None:
            print(f"Code {code} not found in code_to_region mapping.")
            self.debug_check_code(code)
            print("-"*40)
            return None
        if main_region not in self.edited_hierarchy:
            print(f"Main region {main_region} for code {code} not found in hierarchy.")
            self.debug_check_code(code)
            print("-"*40)
            return None
        return self.edited_hierarchy[main_region]["name"]

    def get_code_info(self, code):
        main = self.code_to_region.get(code)
        sub = self.code_to_subregion.get(code)
        if not main or not sub or main not in self.edited_hierarchy or sub not in self.edited_hierarchy[main]["subregions"]:
            return None
        main_name = self.edited_hierarchy[main]["name"]
        sub_name = self.edited_hierarchy[main]["subregions"][sub]["name"]
        code_text = next((c["text"] for c in self.edited_hierarchy[main]["subregions"][sub]["codes"] if c["code"] == code), "")
        return main, main_name, sub, sub_name, code_text

    def reset(self):
        """Reset the edited hierarchy to the original hierarchy."""
        self.edited_hierarchy = copy.deepcopy(self.hierarchy)
        self._rebuild_code_to_region()

    def merge_main_regions(self, new_region, regions_to_merge, new_name=None):
        merged_subregions = {}
        for region in regions_to_merge:
            if region not in self.edited_hierarchy:
                print(f"Region {region} not found, skipping.")
                continue
            for subregion, sub_dict in self.edited_hierarchy[region]["subregions"].items():
                if subregion not in merged_subregions:
                    merged_subregions[subregion] = {
                        "name": sub_dict["name"],
                        "codes": []
                    }
                merged_subregions[subregion]["codes"].extend(sub_dict["codes"])
        if not new_name:
            new_name = self.edited_hierarchy[regions_to_merge[0]]["name"] if regions_to_merge else "Merged Region"
        self.edited_hierarchy[new_region] = {
            "name": new_name,
            "subregions": merged_subregions
        }
        for region in regions_to_merge:
            if region in self.edited_hierarchy:
                del self.edited_hierarchy[region]
        self._rebuild_code_to_region()

    def update_region(self, region, new_name):
        if region in self.edited_hierarchy:
            self.edited_hierarchy[region]["name"] = new_name
        else:
            print(f"Region {region} not found.")
        self._rebuild_code_to_region()

    def split_main_region(self, original_region, subregion_map):
        if original_region not in self.edited_hierarchy:
            print(f"Region {original_region} not found.")
            return
        original_subregions = self.edited_hierarchy[original_region]["subregions"]
        for new_region, subregion_list in subregion_map.items():
            new_subregions = {}
            for subregion in subregion_list:
                if subregion in original_subregions:
                    new_subregions[subregion] = original_subregions[subregion]
            new_name = (
                next(iter(new_subregions.values()))["name"]
                if new_subregions else "Split Region"
            )
            self.edited_hierarchy[new_region] = {
                "name": new_name,
                "subregions": new_subregions
            }
        for subregion_list in subregion_map.values():
            for subregion in subregion_list:
                if subregion in original_subregions:
                    del original_subregions[subregion]
        if not original_subregions:
            del self.edited_hierarchy[original_region]
        self._rebuild_code_to_region()

    def list_main_regions(self, edited=False):
        """List only main regions with total number of codes in parentheses."""
        hierarchy = self.edited_hierarchy if edited else self.hierarchy
        for region in sorted(hierarchy.keys()):
            region_name = hierarchy[region]["name"]
            total_codes = sum(len(sub["codes"]) for sub in hierarchy[region]["subregions"].values())
            print(f"{region}: {region_name} ({total_codes})")

    def print_all_regions(self, edited=False, max_examples=None):
        """ Print all main regions, optionally with their subregions and codes."""
        hierarchy = self.edited_hierarchy if edited else self.hierarchy
        for region in sorted(hierarchy.keys()):
            self.print_region(
                region,
                edited=edited,
                max_examples=max_examples,
            )
            print("-" * 40)

    def print_region(self, region, edited=False, max_examples=None):
        """ Print a single region, optionally with subregions and codes."""
        hierarchy = self.edited_hierarchy if edited else self.hierarchy
        if region not in hierarchy:
            print(f"Region {region} not found.")
            return
        region_name = hierarchy[region]["name"]
        print(f"Region:  {region} {region_name}\n")
        for subregion in sorted(hierarchy[region]["subregions"].keys()):
            self.print_subregion(
                region,
                subregion,
                edited=edited,
                max_examples=max_examples,
            )

    def print_subregion(self, region, subregion, edited=False, max_examples=None):
        """ Print a single subregion and optionally its codes."""
        hierarchy = self.edited_hierarchy if edited else self.hierarchy
        subregions = hierarchy[region]["subregions"]
        if subregion not in subregions:
            print(f"  Subregion {subregion} not found in region {region}.")
            return
        subregion_name = subregions[subregion]["name"]
        codes = subregions[subregion]["codes"]
        print(f"{subregion}: {subregion_name} ({len(codes)} codes)")
        for i, code_info in enumerate(codes):
            if max_examples is not None and i >= max_examples:
                print(f"    ... and {len(codes) - max_examples} more codes")
                break
            print(f"    {code_info['code']}: {code_info['text']}")
        print("")

    def debug_check_code(self, code):
        """ Debug function to check if a code exists in various structures. """
        print("In DataFrame:", code in set(self.codes['SKSkode']))
        print("In code_to_region:", code in self.code_to_region)
        print("In edited hierarchy:", any(
            code in [c["code"] for sub in self.edited_hierarchy[region]["subregions"].values() for c in sub["codes"]]
            for region in self.edited_hierarchy
        ))
        print("In original hierarchy:", any(
            code in [c["code"] for sub in self.hierarchy[region]["subregions"].values() for c in sub["codes"]]
            for region in self.hierarchy
        ))