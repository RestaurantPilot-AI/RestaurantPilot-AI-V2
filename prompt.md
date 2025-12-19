In src\processing\menu_ingestion.py uses this method from src.processing.categorization import get_line_item_category

You will make a new method in src\processing\categorization.py that will be specific for menu item not reusing lin items, that method should use existing methods 
use by line items but prompt should be different specific to that and don't remove any existing method.

this is the file src\storage\database.py containing this method def save_menu_db(menu_df) -> Dict[str, Any]:
this method has some elgacy bersion and stuff nt it shoul d only have one signaute
this:
def save_menu_db(menu_df) -> Dict[str, Any]:
    """Save menu items into CSV-backed 'menu_items' table.

    Usage:
      - save_menu_db(menu_df)  # df contains restaurant_id, menu_item, price, category

    """

    No nee for lgacy support just support this I habve partially updated
 update signature so it just etursn succefully added and if faile raise error dont supress it