Aim: to remove data\database\buildsheet_regex_templates.csv and its need. Right now in src\processing\buildsheet_ingestion.py what we are doing is checking for restuarant id if in our database (data\database\buildsheet_regex_templates.csv) we will get the regex apply on text and save that, 
But if if not in our database we call llm prompt 1 get ground truth then pass that to prompt 2 to get rex apply regex. That is our current setup.
We need to update it to not sue regex at any stage. Whenever we get call we will to prompt no need to check regex just pass the llm cll to further stages and pass it.(Make sure not to change input and output)
In short there is no need for regex so remove apply regex prompt 2

Update the following files src\processing\buildsheet_ingestion.py to make this changes and update src\storage\db_init.py and src\storage\database.py to remove references to buildsheet_regex_templates