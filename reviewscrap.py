from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException

def scrape_all_reviews(product_url):
    # Initialize Chrome WebDriver
    driver = webdriver.Chrome()

    # Open the Amazon product page
    driver.get(product_url)

    # Find and click the "See all reviews" link
    see_all_reviews_link = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[data-hook="see-all-reviews-link-foot"]'))
    )
    see_all_reviews_link.click()

    # Wait for the reviews page to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'cm_cr-review_list'))
    )

    # Array to store reviews
    reviews_array = []

    # Scrape reviews from all pages
    while True:
        # Find all the review elements on the current page
        reviews = driver.find_elements(By.CSS_SELECTOR, '.review-text')
        
        # Store the reviews from the current page in the array
        for review in reviews:
            reviews_array.append(review.text)

        try:
            # Check if there's a "Next page" button
            next_page_button = driver.find_element(By.CSS_SELECTOR, '.a-last > a')
            if 'a-disabled' in next_page_button.get_attribute('class'):
                break  # No more pages to load

            # Click the "Next page" button
            next_page_button.click()

            # Wait for the next page to load
            WebDriverWait(driver, 10).until(
                EC.staleness_of(reviews[0])
            )
        except NoSuchElementException:
            print("No more pages of reviews to load.")
            break
        except StaleElementReferenceException:
            print("StaleElementReferenceException occurred. Retrying...")
            continue

    # Close the WebDriver
    driver.quit()

    return reviews_array

# # URL of the Amazon product page
# product_url = "https://www.amazon.in/gp/product/B07KSP3KT1/ref=ewc_pr_img_3?smid=A15APWRK6P7LBV&psc=1"  # Example URL, replace with the actual product URL

# # Call the function to scrape all reviews
# reviews = scrape_all_reviews(product_url)

# # Iterate over the reviews array to access one review at a time
# print(reviews[0])
