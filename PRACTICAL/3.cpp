#include <iostream>
#include <string>
using namespace std;

// Node Declaration
struct node {
    string label;
    int ch_count;
    struct node* child[10];
} *root;

// Class Declaration
class GT {
public:
    void create_tree();
    void display(node* r1);

    GT() {
        root = NULL;
    }
};

// Create the hierarchical tree
void GT::create_tree() {
    int tchapters;
    root = new node;

    cout << "Enter name of book: ";
    cin.ignore();  // To consume any leftover newline
    getline(cin, root->label);

    cout << "Enter number of chapters in book: ";
    cin >> tchapters;
    root->ch_count = tchapters;

    for (int i = 0; i < tchapters; i++) {
        root->child[i] = new node;

        cout << "Enter the name of Chapter " << i + 1 << ": ";
        cin.ignore();
        getline(cin, root->child[i]->label);

        cout << "Enter number of sections in Chapter " << root->child[i]->label << ": ";
        cin >> root->child[i]->ch_count;

        for (int j = 0; j < root->child[i]->ch_count; j++) {
            root->child[i]->child[j] = new node;
            cout << "Enter Name of Section " << j + 1 << ": ";
            cin.ignore();
            getline(cin, root->child[i]->child[j]->label);
        }
    }
}

// Display the tree
void GT::display(node* r1) {
    if (r1 != NULL) {
        cout << "\n----- Book Hierarchy -----\n";
        cout << "Book Title: " << r1->label << "\n";

        for (int i = 0; i < r1->ch_count; i++) {
            cout << "  Chapter " << i + 1 << ": " << r1->child[i]->label << "\n";
            cout << "    Sections:\n";

            for (int j = 0; j < r1->child[i]->ch_count; j++) {
                cout << "      - " << r1->child[i]->child[j]->label << "\n";
            }
        }
    }
}

int main() {
    int choice;
    GT gt;

    while (true) {
        cout << "\n-----------------\n";
        cout << "Book Tree Creation\n";
        cout << "-----------------\n";
        cout << "1. Create\n";
        cout << "2. Display\n";
        cout << "3. Quit\n";
        cout << "Enter your choice: ";
        cin >> choice;

        switch (choice) {
            case 1:
                gt.create_tree();
                break;  // ✅ Important fix
            case 2:
                gt.display(root);
                break;
            case 3:
                cout << "Thanks for using this program!\n";
                exit(0);
            default:
                cout << "Invalid choice! Please try again.\n";
        }
    }

    return 0;
}   
