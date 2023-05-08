from common.multiapp import MultiApp
from main_page import analyzer, readme_page, visualize, analyzer_for_multiple_molecules


main = MultiApp()
main.add_app("DPPH analyzer", analyzer.predict_dpph)
main.add_app('DPPH analyzer for multiple molecules', analyzer_for_multiple_molecules.predict_dpph_multi)
main.add_app("visualize", visualize.visualize)
main.add_app("README", readme_page.explain)

main.run()