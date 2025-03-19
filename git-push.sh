# message: Update as parameter
# Check if there is a parameter
if [ -z "$2" ]
  then
    echo "No commit message supplied"
    exit 1
fi
echo "Add ..."
git add .
echo "Commit ..."
git commit -m "$2"
echo "Push ..."
git push origin $1