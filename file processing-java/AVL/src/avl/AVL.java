package avl;
//실행환경 JAVA ECLIPSE를 22-09를 통해 작성하였음 // AVL Tree관련된것들은 BST아래에 분리해서 작성하였음.
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

final class Main {
	public static void main(String args[]) {
		AVL tree = new AVL(); // AVL트리를 생성.
		File file = new File("src/avl/AVL-input.txt");//txt파일 읽기
		try (BufferedReader br = new BufferedReader(new FileReader(file))) { // 버퍼를 생성해서 파일을 읽어오고.
		    String line;
		    while ((line = br.readLine()) != null) { // txt파일로부터 내용이 없을떄까지 한줄씩 line에 저장.
		    	String[] command = new String[2]; // 2개짜리 배열인 command에 저장후
		    	command = line.split(" ");//split을 통해서 공백을 기준으로 저장함
		    	if(command[0].compareTo("i") == 0) { // i일경우인풋이기때문에 insertAVL를 호출하고 아닐경우엔 deleteAVL를 호출함.
		    		
		    		tree.insertAVL(tree, Integer.parseInt(command[1]));
		    		System.out.println();
		    	}
		    	else {
		    		tree.deleteAVL(tree, Integer.parseInt(command[1]));
		    	}
		    }
		} catch (IOException e) {
		    e.printStackTrace(); // Exception 처리
		}
	}
}
abstract interface INode { // Node 인터페이스 부분, interface를 통해 Node Class에 있어야 할 함수들을 정의.
	
	public int element(); 
	
	public int height();
	
	public Node left();
	
	public Node right();
	
	public boolean noNodes();

}

final class Node implements INode{ //Node 부분.
	int element;
	int height;
	Node left;
	Node right;
	int balanceF;
	public int element() {
		return this.element;
	}
	public int height() {
		return this.height;
	}
	public Node left() {
		return this.left;
	}
	public Node right() {
		return this.right;
	}
	public boolean noNodes() { // Node class의 noNodes는 Child가 없을 경우 true를 있을경우 false를 반환.
		if (this.left == null && this.right == null) {
			return true;
		}
		else {
			return false;
		}
		
	}
	public int getHeight(Node root) { // subtree의 Height를 구하는 함수.
		int leftHeight = 0;
        int rightHeight = 0;

        if (root.left != null) {
            leftHeight = 1 + getHeight(root.left);
        }

        if (root.right != null) {
            rightHeight = 1 + getHeight(root.right);
        }

        if( leftHeight > rightHeight ) {
        	this.height = leftHeight;
        	return leftHeight;
        }
        else {
        	this.height = rightHeight;
        	return rightHeight;
        }
	}
	public Node(int element) {
		
		this.element = element;
	}
}

public class AVL {
	Node root = null;
	
	public AVL() { // 생성자
		this.root = null;
	}
	public Node getRoot() { //root반환하는 getter
		return this.root;
	}
	public int height(Node T) { // Node T 를 기준으로 subtree의 height를 반환하는 함수.
		if(T == null) {
			return 0;
		}
		else {
			int leftHeight = height(T.left());
			int rightHeight = height(T.right());
		
			if(leftHeight > rightHeight) {
				return leftHeight + 1;
			}
			else {
				return rightHeight + 1;
			}
		
    
		}	
	}
	
	public void insertBST(AVL T, int insertKey) { //insert를 구현하는 함수.
		Node parents = null;
		Node now = T.getRoot(); // 현재 보고있는Node를 now로 정의.
		
		while(now!= null) {
			if(insertKey == now.element()) {
				System.out.println("i" + " "+ insertKey + " : The key already exists");//insert하려는 key가 존재하면 print후 return.
				return;
			}
			parents = now; // parents에 now를 저장하고 now를 한단계 내릴 준비를함.
			if(insertKey < now.element()) { // key가 현재 element보다 작으면 왼쪽child로 now를 이동.
				now = now.left();
			}
			else {
				now = now.right(); // 크면 오른쪽으로 이동.
			}
		}
		Node newNode = new Node(insertKey); // 넣을node를 생성하고.
		if(T.getRoot() == null) { // Root가 없는 null BST면 루트에 저장.
			this.root = newNode;
		}
		else if(insertKey < parents.element()) { // 작으면 왼쪽 크면 오른쪽에 배치.
			parents.left = newNode;
		}
		else {
			parents.right = newNode;
		}
		newNode.balanceF  = height(newNode.left()) - height(newNode.right());
	}
	
	public void visit(Node T) { // visit을 통해 element를 출력.
		System.out.print("("+ T.element() + ", " + T.balanceF+")"+" ");
	}
	
	public void inOrder(Node T) { // inorder순서에따라서 재귀적으로 element출력.
		if (T == null) {
			return;
		}
		inOrder(T.left());
		T.balanceF  = height(T.left()) - height(T.right());
		visit(T);
		inOrder(T.right());
	}
	
	public int noNodes(Node T) { // noNodes함수를 통해 서브트리의 child개수를 구함.
		int no = 0;
		if(T == null) {
			return 0;
		}
		no = no+1;
		no = noNodes(T.left());
		no = noNodes(T.right());
		return no;
	}
	
	public Node search(Node T, int searchKey) { // search함수를 통해 searchKey를 가진 node가 존재하는지를 검색.
		Node parents = null;
		Node now = T;
		while(now != null) {
			if(searchKey == now.element()) {
				return now;
			}
			parents = now;
			if(searchKey < now.element()) {
				now = now.left();
			}
			else {
				now = now.right();
			}
		}
		return null;
	}
	
	public Node getParents(Node T, int key) { // getParents를 통해 부모Node가 존재하는지를 검사 있으면 parents를 없으면 null을 return.
		Node parents = null;
		Node now = T;
		if(T == null) {
			return null;
		}
		while(now != null) {
			if(key == now.element()) {
				return parents;
			}
			parents = now;
			if(key < now.element()) {
				now = now.left();
			}
			else {
				now = now.right();
			}
		}
		return null;
	}
	public Node minNode(Node T) { // subtree의 노드들중 가장 작은 node를 return.
		if(T.left() == null) {
			return T;
		}
		else {
			return minNode(T.left());
		}
	}
	public Node maxNode(Node T) { // subtree의 노드들중 가장 큰 node를 return.
		if(T.right() == null) {
			return T;
		}
		else {
			return maxNode(T.right());
		}
	}
	public Node deleteBST(Node T, int deleteKey) { // 삭제하는 부분. 삭제후 상황에 맞게 null혹은 삭제한 노드와 바뀐Node의 부모Node를 반환
		Node parents = getParents(T, deleteKey);
		Node p = search(T, deleteKey);
		if(p == null) { //search함수를 통해 삭제하려는 Node가 없으면 print후 return null;
			System.out.println("d " + deleteKey + " : The key does not exist");
			this.inOrder(root);
			System.out.println();
			return null;
		}
		if(parents == null && p.left() == null && p.right() == null) { // 삭제하려는 Node가 루트일경우 루트 삭제후 return null.
			this.root = null;
			System.out.print("NO ");
			return null;
		}
		if(p.left() == null && p.right() == null) { // Child가 없는 경우.
			if(parents.left() == p) { // 삭제하려는노드가 부모노드의 좌측일경우. 부모노드를 반환.
				parents.left = null;
				return parents;
				
			}
			else {
				parents.right = null; // 우측일경우. 부모노드를 반환.
				return parents;
				
			}
		}
		else if(parents == null && ((p.left() == null && p.right() != null) || (p.left() != null && p.right() == null))) { // parents가 없고 p의 자식이 하나인경우.
			if(p.left() == null) { // 부모가 없기떄문에 자식을 반환.
				root = p.right();
				return p.right();
			}
			else {
				root = p.left(); // 부모가 없기떄문에 자식을 반환.
				return p.right();
			}
		}
		else if(parents != null && (p.left() == null && p.right() != null) || (p.left() != null && p.right() == null)) { // parents가 있고 p의 자식이 하나인경우.
			if(p.left == null) { // 삭제하려는 노드의 우측child만있는경우.
				if(parents.left == p) {
					parents.left = p.right();// 부모가 없기떄문에 자신을 반환.
					return p.right();
					
				}
				else {
					parents.right = p.right();
					return p.right(); // 부모가 없기떄문에 자신을 반환.
				}
				
			}
			else {		//삭제하려는 노드의 좌측child만있는경우.
				if(parents.left == p) {
					parents.left = p.left();
					return p.left(); // 부모가 없기떄문에 자신을 반환.

					
				}
				else {
					parents.right = p.left();
					return p.left(); // 부모가 없기떄문에 자신을 반환.
				
				}	
			}
		}
		else if(p.left() != null && p.right() != null) { // 삭제하려는 Node의 child가 좌우 다 있는경우.
			int flag = 0; // 0 = left 1 = right; // flag를 통해 좌인지 우인지를 판단하고.
			Node r = null;
			Node returnNode = null;
			if(height(p.left()) > height(p.right())) { // p좌측의 높이가 우측의 높이보다 크면.
				flag = 0;
				r = maxNode(p.left()); //좌측에서 빼야하니 r을 좌측의 노드중 가장 큰 노드로 설정하기위해 maxNode호출.
				returnNode = getParents(this.root, r.element); //returnNode를 통해서 삭제요청들어온 노드와 바뀐노드의 부모를 리턴.
			}
			if(height(p.left()) < height(p.right())) { // 우측의 높이가 크면
				
				flag = 1;
				r = minNode(p.right()); //우측에서 빼야하니 r 우측의 subtree중 가장 작은것을 호출하기위해 minNode호출.
				returnNode = getParents(this.root, r.element);//returnNode를 통해서 삭제요청들어온 노드와 바뀐노드의 부모를 리턴.
			}
			if(height(p.left()) == height(p.right())) { // 좌측과 우측의subtree가 같으면
				if(noNodes(p.left()) > noNodes(p.right())){ // noNodes를 호출해 개수를 비교해서 왼쪽 오른쪽을 판단함.
					
					r = maxNode(p.left());
					flag = 0;
					returnNode = getParents(this.root, r.element);//returnNode를 통해서 삭제요청들어온 노드와 바뀐노드의 부모를 리턴.
				}
				else {
					
					r = minNode(p.right());
					flag = 1;
					returnNode = getParents(this.root, r.element);//returnNode를 통해서 삭제요청들어온 노드와 바뀐노드의 부모를 리턴.
				}
			}
			
			
			if(flag == 0) { // flag가 0 == 좌측일 경우와 우측일경우 모두 p를 r로 대치함.
				
				deleteBST(p, r.element());
				p.element = r.element;
				return returnNode;//returnNode를 통해서 삭제요청들어온 노드와 바뀐노드의 부모를 리턴.
			}
			else {
				
				deleteBST(p, r.element());
				p.element = r.element;
				return returnNode;//returnNode를 통해서 삭제요청들어온 노드와 바뀐노드의 부모를 리턴.
			}
		}
		return null; //Node return을 위한 null
	}
	///////////////////////////////////////////////////////AVL TREE BELOW////////////////////////////////////////////////////////////////////////////
	public void insertAVL(AVL T, int newKey) { // insertAVL
		if(search(T.getRoot(), newKey) != null){ // 추가하려는 키가 이미있으면 insertBST만 실행후 종료
			insertBST(T, newKey);
			this.inOrder(T.getRoot());
		}
		else { // 키가 없으면 먼저 추가한후에 추가한 키를 기준으로 checkBalance를 돌린후 inOrder를 출력.
			insertBST(T, newKey);
			Node p = search(T.getRoot(), newKey);
			checkBalance(T.getRoot(), p, null, newKey);
			this.inOrder(T.getRoot());
		}
	}
	public void deleteAVL(AVL T, int newKey) { // deleteAVL
		Node p = deleteBST(T.getRoot(), newKey); // 먼저P를 지움.
		if (p == null) { // 그뒤 deleteBST에서 null값이 반환된 경우는 회전이 없는경우이므로 그대로 종료.
			return;
		}
		else {
			checkBalance(T.getRoot(), p, null, p.element()); // return이 있는경우에는 return된 Node를 중심으로 checkBalance후 inorder출력.
			this.inOrder(T.getRoot());
			System.out.println();
		}
	}
	
	public void checkBalance(Node T, Node p, Node q, Integer newKey) {
		int rotateType = 0;//type을 결정 1 = LL, 2 = LR, 3 = RR, 4 = RL
		while(p!= null) {
			p.balanceF  = height(p.left()) - height(p.right());//받은 Node p의 balanceFactor를 계산후.
			if(p.balanceF == 2) {
				p.left().balanceF = height(p.left().left()) - height(p.left().right());//p의 balanceF가 2이면 p의 좌측노드의 balanceF를 계산.
				if(p.left().balanceF >= 0) { // p좌측의 BF가 0보다 크거나 같으면 LL변환 >=대신 >해도되는데 그러면 정상작동하지만 예제와 결과가 달라짐.
						//LL
					rotateType = 1;
					break;
				}
				else { // p좌측의 balanceF가 0보다 작을때.
					//LR
					rotateType = 2;
					break;
				}
			}
			else if(p.balanceF == -2){//p의 balanceF가 -2이면 p의 우측노드의 balanceF를 계산.
				p.right().balanceF = height(p.right().left()) - height(p.right().right());
				if(p.right().balanceF <= 0) { // p우측의 BF가 0보다 작거나 같으면 RR변환 <=대신 <해도되는데 그러면 정상작동하지만 예제와 결과가 달라짐.
						//RR
					rotateType = 3;
					break;
				}
				else {// p우측의 balanceF가 0보다 클때.
						//RL
					rotateType = 4;
					break;
				}
			}
			p = getParents(T, p.element()); // 조건에 맞지 않으면 p의 부모로 올라가서 재탐색.
		}
			
		/*while(p!= null) { // 실패한 알고리즘 key값으로 정렬했으나 delete에서 키값설정이 안먹혀 실패.
			p.balanceF  = height(p.left()) - height(p.right());
			if(p.balanceF == -2 || p.balanceF == 2) {
				
				if(newKey < p.element()) {
					if(newKey < p.left().element()) {
						//LL
						rotateType = 1;
						break;
					}
					else {
						//LR
						rotateType = 2;
						break;
					}
				}
				else {
					if(newKey > p.right().element()) {
						//RR
						rotateType = 3;
						break;
					}
					else {
						//RL
						rotateType = 4;
						break;
					}
				}
				
			}
			p = getParents(T, p.element());
		}*/
		if(rotateType != 0) { // Rotation필요하면 rotateTree호출.
			rotateTree(T, rotateType, p, q);
		}
		
		else {//Rotation 필요없는경우.
			System.out.print("NO ");
		}
		
	}
	
	public void rotateTree(Node T, int rotateType, Node p, Node q) {//Tree를 받아온 Type에 맞게 돌리는 함수.
		q = getParents(T, p.element()); // 부모 받아오고.
		if(q == null) { // 부모 없을때
			if(rotateType == 1) { // LL일때 노드 포인터 변환후 출력.
				root =p.left();
				Node base = root;
				Node temp = p.left().right;
				base.left = p.left().left;
				base.right = p;
				p.left = temp;
				System.out.print("LL ");
			}
			else if(rotateType == 2) { // LL일때 노드 포인터 변환후 출력.
				root = p.left().right;
				Node base = root;
				Node templeft = base.left;
				Node tempright = base.right;
				base.left = p.left();
				base.right = p;
				base.left().right = templeft;
				base.right().left = tempright;
				System.out.print("LR ");
			}
			else if(rotateType == 3){// RR일때 노드 포인터 변환후 출력.
				root = p.right();
				Node base = root;
				Node temp = p.right().left;
				base.right = p.right().right();
				base.left = p;
				p.right = temp;
				System.out.print("RR ");
			}
			else { // RL일때 노드 포인터 변환후 출력.
				root = p.right().left;
				Node base = root;
				Node templeft = base.left;
				Node tempright = base.right;
				base.right = p.right();
				base.left = p;
				base.left.right = templeft;
				base.right.left = tempright;
				System.out.print("RL ");
			}
		}
		
		else { // 부모있을때
			if(rotateType == 1) { // LL일때 노드 포인터 변환후 출력.
				
				Node base = p.left();
				Node temp = p.left().right;
				base.left = p.left().left;
				base.right = p;
				p.left = temp;
				if(q.left().element() == p.element()) { //코드재활용을 위한 if else구조 LL은 left RR은 right만 있어도됨.
					q.left = base;
				}
				else {
					q.right = base;
				}
				System.out.print("LL ");
				
			}
			else if(rotateType == 2) { // LR일때 노드 포인터 변환후 출력.
				
				Node base = p.left().right;
				Node templeft = base.left;
				Node tempright = base.right;
				base.left = p.left();
				base.right = p;
				base.left().right = templeft;
				base.right().left = tempright;
				if(q.left().element() == p.element()) {
					q.left = base;
				}
				else {
					q.right = base;
				}
				System.out.print("LR ");
			}
			else if(rotateType == 3){ // RR일때 노드 포인터 변환후 출력.
				
				Node base = p.right();
				Node temp = p.right().left;
				base.right = p.right().right();
				base.left = p;
				p.right = temp;
				if(q.left().element() == p.element()) {
					q.left = base;
				}
				else {
					q.right = base;
				}
				System.out.print("RR ");
			}
			else { // RL일때 노드 포인터 변환후 출력.
				
				Node base = p.right.left;
				Node templeft = base.left;
				Node tempright = base.right;
				base.right = p.right();
				base.left = p;
				base.left.right = templeft;
				base.right.left = tempright;
				if(q.left().element() == p.element()) {
					q.left = base;
				}
				else {
					q.right = base;
				}
				System.out.print("RL ");
			}
		}
	}
	
}



